"""Annotation documents describing the analysed differences of a comparison.

An annotation document is a JSON file written by the analysts, by hand or
through the in-app editor. It carries the report metadata, a set of
categories (e.g. *action required*, *expected difference*), an ordered
outline of sections, and the annotations themselves. Each annotation
points at one or more variables (``fnmatch`` globs allowed) and may list
figures to render with a given filter expression and zoom range.

The document is consumed by the Dash app (badges, detail panels, editor)
and by :mod:`app.annotated_report` to produce a standalone report.

Command line::

    python -m app.annotations schema
    python -m app.annotations template ref.nc tgt.nc -o annotations.json
    python -m app.annotations check annotations.json [ref.nc tgt.nc]
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import fnmatch
import hashlib
import json
import logging
import os
import pathlib
import sys
import threading
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

from .constants import DEFAULT_TOLERANCE_STEPS, PLOT_LABELS

if TYPE_CHECKING:
    from .comparison import GroupComparisonResult

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: Allowed values of :attr:`Annotation.status`.
STATUSES = ("open", "confirmed", "resolved", "wontfix")

#: Languages available for the fixed texts of the generated report.
LANGUAGES = ("en", "fr")

#: Allowed values of :attr:`Section.auto`.
AUTO_SECTIONS = ("agreement_table", "unannotated", "summary_counts")

#: Variable statuses considered as "a difference to explain".
DIFFERENT_STATUSES = frozenset(
    {"different", "almost_equal", "missing_in_source", "missing_in_target"}
)

_GLOB_CHARS = frozenset("*?[")


class AnnotationError(ValueError):
    """Raised when an annotation document is structurally invalid."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


class ConflictError(RuntimeError):
    """Raised when saving would overwrite changes made elsewhere."""


class ReadOnlyError(RuntimeError):
    """Raised when mutating a store opened read-only."""


@dataclasses.dataclass
class DatasetRow:
    """One line of the "compared data" table of the report."""

    label: str = ""
    value: str = ""


@dataclasses.dataclass
class Category:
    """A family of annotations, rendered with a colour badge."""

    id: str
    label: str = ""
    color: str = "#607d8b"
    description: str = ""


@dataclasses.dataclass
class FigureSpec:
    """A figure to render next to an annotation.

    Attributes:
        variable: Variable path to plot.
        plots: Figure labels to keep (see ``PLOT_LABELS``); empty means all.
        expression: Optional filter expression (``${/path} > 0`` syntax).
        zoom: Optional ``[xmin, xmax]`` range applied to the index axis.
        caption: Markdown caption.

    """

    variable: str
    plots: list[str] = dataclasses.field(default_factory=list)
    expression: str = ""
    zoom: list[float] | None = None
    caption: str = ""


@dataclasses.dataclass
class Annotation:
    """An analysed difference.

    Attributes:
        id: Unique identifier, used as anchor and in section references.
        title: Short title.
        category: Identifier of a :class:`Category`.
        status: One of ``STATUSES``.
        variables: Variable paths or ``fnmatch`` globs.
        text: Markdown description of the observation.
        effect: Markdown description of the consequence.
        magnitude: Free-form magnitude ("69 measurements, 3.4 %").
        cause: Free-form cause.
        refs: External references (tickets, specification items).
        figures: Figures to render.

    """

    id: str
    title: str = ""
    category: str = ""
    status: str = "open"
    variables: list[str] = dataclasses.field(default_factory=list)
    text: str = ""
    effect: str = ""
    magnitude: str = ""
    cause: str = ""
    refs: list[str] = dataclasses.field(default_factory=list)
    figures: list[FigureSpec] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Section:
    """A node of the report outline."""

    id: str
    title: str = ""
    body: str = ""
    annotations: list[str] = dataclasses.field(default_factory=list)
    auto: str | None = None
    subsections: list[Section] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class AnnotationDocument:
    """Root of an annotation file."""

    schema_version: int = SCHEMA_VERSION
    language: str = "en"
    title: str = ""
    subtitle: str = ""
    date: str = ""
    authors: list[str] = dataclasses.field(default_factory=list)
    source_label: str = "Source"
    target_label: str = "Target"
    datasets: list[DatasetRow] = dataclasses.field(default_factory=list)
    categories: list[Category] = dataclasses.field(default_factory=list)
    sections: list[Section] = dataclasses.field(default_factory=list)
    annotations: list[Annotation] = dataclasses.field(default_factory=list)

    def category(self, category_id: str) -> Category | None:
        """Return the category with the given id, if any."""
        return next((c for c in self.categories if c.id == category_id), None)

    def annotation(self, annotation_id: str) -> Annotation | None:
        """Return the annotation with the given id, if any."""
        return next((a for a in self.annotations if a.id == annotation_id), None)

    def iter_sections(self) -> Iterator[tuple[Section, Section | None, int]]:
        """Yield ``(section, parent, depth)`` in document order."""

        def _walk(
            sections: list[Section], parent: Section | None, depth: int
        ) -> Iterator[tuple[Section, Section | None, int]]:
            for section in sections:
                yield section, parent, depth
                yield from _walk(section.subsections, section, depth + 1)

        yield from _walk(self.sections, None, 0)

    def section(self, section_id: str) -> Section | None:
        """Return the section with the given id, if any."""
        return next((s for s, _, _ in self.iter_sections() if s.id == section_id), None)

    def unplaced_annotations(self) -> list[Annotation]:
        """Annotations not referenced by any section, in document order."""
        placed = {aid for s, _, _ in self.iter_sections() for aid in s.annotations}
        return [a for a in self.annotations if a.id not in placed]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain JSON-compatible data, dropping empty optionals."""
        data: dict[str, Any] = _prune(dataclasses.asdict(self))
        return data


def default_categories() -> list[Category]:
    """Categories proposed for a new document."""
    return [
        Category("action", "Action required", "#c62828"),
        Category("expected", "Expected difference", "#1565c0"),
        Category("fixed", "Defect fixed", "#2e7d32"),
        Category("question", "Open question", "#6a1b9a"),
    ]


def new_document() -> AnnotationDocument:
    """Return an empty document with the default categories."""
    return AnnotationDocument(categories=default_categories())


# ── (De)serialisation ───────────────────────────────────────────────


def _prune(value: Any) -> Any:
    """Drop ``None``/empty values from dicts so saved files stay compact.

    Identifiers and titles are kept even when empty so the structure of
    an entry stays readable.
    """
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            pruned = _prune(v)
            if pruned in (None, "", []) and k not in {"id", "title", "variable"}:
                continue
            out[k] = pruned
        return out
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


def _str(raw: dict[str, Any], key: str, where: str, errors: list[str]) -> str:
    value = raw.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        errors.append(f"{where}.{key}: expected a string")
        return str(value)
    return value


def _str_list(
    raw: dict[str, Any], key: str, where: str, errors: list[str]
) -> list[str]:
    value = raw.get(key, [])
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        errors.append(f"{where}.{key}: expected a list of strings")
        return []
    return list(value)


def _obj_list(
    raw: dict[str, Any], key: str, where: str, errors: list[str]
) -> list[Any]:
    value = raw.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f"{where}.{key}: expected a list")
        return []
    out = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"{where}.{key}[{i}]: expected an object")
            continue
        out.append(item)
    return out


def _parse_figure(raw: dict[str, Any], where: str, errors: list[str]) -> FigureSpec:
    zoom = raw.get("zoom")
    if zoom is not None:
        if (
            not isinstance(zoom, list)
            or len(zoom) != 2
            or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in zoom
            )
        ):
            errors.append(f"{where}.zoom: expected [xmin, xmax]")
            zoom = None
        else:
            zoom = [float(zoom[0]), float(zoom[1])]
    return FigureSpec(
        variable=_str(raw, "variable", where, errors),
        plots=_str_list(raw, "plots", where, errors),
        expression=_str(raw, "expression", where, errors),
        zoom=zoom,
        caption=_str(raw, "caption", where, errors),
    )


def _parse_annotation(raw: dict[str, Any], where: str, errors: list[str]) -> Annotation:
    return Annotation(
        id=_str(raw, "id", where, errors),
        title=_str(raw, "title", where, errors),
        category=_str(raw, "category", where, errors),
        status=_str(raw, "status", where, errors) or "open",
        variables=_str_list(raw, "variables", where, errors),
        text=_str(raw, "text", where, errors),
        effect=_str(raw, "effect", where, errors),
        magnitude=_str(raw, "magnitude", where, errors),
        cause=_str(raw, "cause", where, errors),
        refs=_str_list(raw, "refs", where, errors),
        figures=[
            _parse_figure(f, f"{where}.figures[{i}]", errors)
            for i, f in enumerate(_obj_list(raw, "figures", where, errors))
        ],
    )


def _parse_section(raw: dict[str, Any], where: str, errors: list[str]) -> Section:
    auto = raw.get("auto")
    if auto is not None and not isinstance(auto, str):
        errors.append(f"{where}.auto: expected a string")
        auto = None
    return Section(
        id=_str(raw, "id", where, errors),
        title=_str(raw, "title", where, errors),
        body=_str(raw, "body", where, errors),
        annotations=_str_list(raw, "annotations", where, errors),
        auto=auto or None,
        subsections=[
            _parse_section(s, f"{where}.subsections[{i}]", errors)
            for i, s in enumerate(_obj_list(raw, "subsections", where, errors))
        ],
    )


def document_from_dict(raw: Any) -> AnnotationDocument:
    """Build and validate a document from parsed JSON.

    Unknown keys are ignored so files written by a newer version stay
    loadable.

    Raises:
        AnnotationError: If the structure or the cross references are invalid.

    """
    if not isinstance(raw, dict):
        raise AnnotationError(["document: expected a JSON object"])
    errors: list[str] = []
    version = raw.get("schema_version", SCHEMA_VERSION)
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        errors.append(
            f"schema_version: unsupported value {version!r} "
            f"(this version reads up to {SCHEMA_VERSION})"
        )
    doc = AnnotationDocument(
        schema_version=SCHEMA_VERSION,
        language=_str(raw, "language", "document", errors) or "en",
        title=_str(raw, "title", "document", errors),
        subtitle=_str(raw, "subtitle", "document", errors),
        date=_str(raw, "date", "document", errors),
        authors=_str_list(raw, "authors", "document", errors),
        source_label=_str(raw, "source_label", "document", errors) or "Source",
        target_label=_str(raw, "target_label", "document", errors) or "Target",
        datasets=[
            DatasetRow(
                label=_str(d, "label", f"datasets[{i}]", errors),
                value=_str(d, "value", f"datasets[{i}]", errors),
            )
            for i, d in enumerate(_obj_list(raw, "datasets", "document", errors))
        ],
        categories=[
            Category(
                id=_str(c, "id", f"categories[{i}]", errors),
                label=_str(c, "label", f"categories[{i}]", errors),
                color=_str(c, "color", f"categories[{i}]", errors) or "#607d8b",
                description=_str(c, "description", f"categories[{i}]", errors),
            )
            for i, c in enumerate(_obj_list(raw, "categories", "document", errors))
        ],
        sections=[
            _parse_section(s, f"sections[{i}]", errors)
            for i, s in enumerate(_obj_list(raw, "sections", "document", errors))
        ],
        annotations=[
            _parse_annotation(a, f"annotations[{i}]", errors)
            for i, a in enumerate(_obj_list(raw, "annotations", "document", errors))
        ],
    )
    errors.extend(validate(doc))
    if errors:
        raise AnnotationError(errors)
    return doc


def validate(doc: AnnotationDocument) -> list[str]:
    """Check identifiers and cross references; return the error messages."""
    errors: list[str] = []
    if doc.language not in LANGUAGES:
        errors.append(f"language: {doc.language!r} not in {', '.join(LANGUAGES)}")

    def _check_ids(kind: str, ids: Iterable[str]) -> None:
        seen: set[str] = set()
        for ident in ids:
            if not ident.strip():
                errors.append(f"{kind}: empty id")
            elif ident in seen:
                errors.append(f"{kind}: duplicate id {ident!r}")
            seen.add(ident)

    _check_ids("categories", (c.id for c in doc.categories))
    _check_ids("annotations", (a.id for a in doc.annotations))
    _check_ids("sections", (s.id for s, _, _ in doc.iter_sections()))

    category_ids = {c.id for c in doc.categories}
    annotation_ids = {a.id for a in doc.annotations}

    for ann in doc.annotations:
        where = f"annotation {ann.id!r}"
        if ann.category and ann.category not in category_ids:
            errors.append(f"{where}: unknown category {ann.category!r}")
        if ann.status not in STATUSES:
            errors.append(
                f"{where}: status {ann.status!r} not in {', '.join(STATUSES)}"
            )
        for fig in ann.figures:
            if not fig.variable.strip():
                errors.append(f"{where}: figure without variable")
            for label in fig.plots:
                if label not in PLOT_LABELS:
                    errors.append(
                        f"{where}: unknown plot {label!r} "
                        f"(expected one of {', '.join(PLOT_LABELS)})"
                    )
            if fig.zoom is not None and fig.zoom[0] >= fig.zoom[1]:
                errors.append(f"{where}: zoom must satisfy xmin < xmax")

    for section, _, _ in doc.iter_sections():
        where = f"section {section.id!r}"
        if section.auto is not None and section.auto not in AUTO_SECTIONS:
            errors.append(
                f"{where}: auto {section.auto!r} not in {', '.join(AUTO_SECTIONS)}"
            )
        for ref in section.annotations:
            if ref not in annotation_ids:
                errors.append(f"{where}: unknown annotation {ref!r}")
    return errors


def load_annotations(path: str | pathlib.Path) -> AnnotationDocument:
    """Read and validate an annotation file.

    Raises:
        AnnotationError: If the file is not valid JSON or fails validation.

    """
    try:
        raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnnotationError([f"{path}: invalid JSON ({exc})"]) from exc
    return document_from_dict(raw)


def dump_annotations(doc: AnnotationDocument) -> str:
    """Serialise a document as stable, diff-friendly JSON text."""
    return json.dumps(doc.to_dict(), indent=2, ensure_ascii=False) + "\n"


# ── Matching against comparison results ─────────────────────────────


def pattern_matches(pattern: str, varpath: str) -> bool:
    """Return True if ``varpath`` matches an exact path or a glob pattern."""
    if _GLOB_CHARS.isdisjoint(pattern):
        return pattern == varpath
    return fnmatch.fnmatchcase(varpath, pattern)


def annotation_covers(annotation: Annotation, varpath: str) -> bool:
    """Return True if one of the annotation's patterns matches ``varpath``."""
    return any(pattern_matches(p, varpath) for p in annotation.variables)


def match_annotations(
    doc: AnnotationDocument, varpaths: Iterable[str]
) -> dict[str, list[Annotation]]:
    """Map every variable path to the annotations covering it.

    Variables without annotation are omitted from the result.
    """
    out: dict[str, list[Annotation]] = {}
    for varpath in varpaths:
        matched = [a for a in doc.annotations if annotation_covers(a, varpath)]
        if matched:
            out[varpath] = matched
    return out


def all_varpaths(results: list[GroupComparisonResult]) -> list[str]:
    """Return every variable path of the comparison results."""
    return [v.varpath for group in results for v in group.variables]


def check_against_results(
    doc: AnnotationDocument, results: list[GroupComparisonResult]
) -> tuple[list[str], list[str]]:
    """Cross-check a document with comparison results.

    Returns:
        ``(warnings, unannotated)``: messages for patterns or figures that
        match no variable, and the paths of differing variables that no
        annotation covers.

    """
    varpaths = all_varpaths(results)
    known = set(varpaths)
    warnings: list[str] = []
    for ann in doc.annotations:
        for pattern in ann.variables:
            if not any(pattern_matches(pattern, v) for v in varpaths):
                warnings.append(
                    f"annotation {ann.id!r}: {pattern!r} matches no variable"
                )
        for fig in ann.figures:
            if fig.variable not in known:
                warnings.append(
                    f"annotation {ann.id!r}: figure variable {fig.variable!r} not found"
                )
    covered = match_annotations(doc, varpaths)
    unannotated = [
        v.varpath
        for group in results
        for v in group.variables
        if v.status in DIFFERENT_STATUSES and v.varpath not in covered
    ]
    return warnings, unannotated


# ── Document edition helpers ────────────────────────────────────────


def upsert_annotation(
    doc: AnnotationDocument,
    annotation: Annotation,
    *,
    previous_id: str | None = None,
    section_id: str | None = None,
) -> None:
    """Insert or replace an annotation.

    ``previous_id`` handles renames: the old entry is replaced in place
    and section references are updated. ``section_id`` attaches the
    annotation to that section if it is not already referenced there.
    """
    old_id = previous_id or annotation.id
    for i, existing in enumerate(doc.annotations):
        if existing.id == old_id:
            doc.annotations[i] = annotation
            break
    else:
        doc.annotations.append(annotation)
    if old_id != annotation.id:
        for section, _, _ in doc.iter_sections():
            section.annotations = [
                annotation.id if a == old_id else a for a in section.annotations
            ]
    if section_id:
        target = doc.section(section_id)
        if target is not None and annotation.id not in target.annotations:
            target.annotations.append(annotation.id)


def delete_annotation(doc: AnnotationDocument, annotation_id: str) -> None:
    """Remove an annotation and every section reference to it."""
    doc.annotations = [a for a in doc.annotations if a.id != annotation_id]
    for section, _, _ in doc.iter_sections():
        section.annotations = [a for a in section.annotations if a != annotation_id]


def _siblings(doc: AnnotationDocument, section_id: str) -> list[Section] | None:
    for section, parent, _ in doc.iter_sections():
        if section.id == section_id:
            return parent.subsections if parent is not None else doc.sections
    return None


def upsert_section(
    doc: AnnotationDocument,
    section: Section,
    *,
    previous_id: str | None = None,
    parent_id: str | None = None,
) -> None:
    """Insert or replace a section.

    An existing section keeps its position and its subsections unless
    the new value provides some. A new section is appended to
    ``parent_id`` (or to the root).
    """
    old_id = previous_id or section.id
    siblings = _siblings(doc, old_id)
    if siblings is not None:
        idx = next(i for i, s in enumerate(siblings) if s.id == old_id)
        if not section.subsections:
            section.subsections = siblings[idx].subsections
        siblings[idx] = section
        return
    parent = doc.section(parent_id) if parent_id else None
    (parent.subsections if parent is not None else doc.sections).append(section)


def delete_section(doc: AnnotationDocument, section_id: str) -> None:
    """Remove a section and its subsections (annotations are kept)."""
    siblings = _siblings(doc, section_id)
    if siblings is not None:
        siblings[:] = [s for s in siblings if s.id != section_id]


def move_section(doc: AnnotationDocument, section_id: str, delta: int) -> None:
    """Move a section ``delta`` positions among its siblings."""
    siblings = _siblings(doc, section_id)
    if siblings is None:
        return
    idx = next(i for i, s in enumerate(siblings) if s.id == section_id)
    new_idx = max(0, min(len(siblings) - 1, idx + delta))
    siblings.insert(new_idx, siblings.pop(idx))


def unique_id(base: str, existing: Iterable[str]) -> str:
    """Return ``base`` or ``base-N`` so that it is not in ``existing``."""
    taken = set(existing)
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in base.lower())
    slug = "-".join(filter(None, slug.split("-"))) or "item"
    candidate, n = slug, 2
    while candidate in taken:
        candidate, n = f"{slug}-{n}", n + 1
    return candidate


# ── Persistent store ────────────────────────────────────────────────


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class AnnotationStore:
    """Process-wide owner of an annotation file.

    Shared by every browser tab. Every change goes through :meth:`apply`,
    which validates the edited copy, writes it atomically (keeping a
    ``.bak`` of the previous content) and bumps :attr:`revision`.

    Two kinds of conflicts are detected: another tab saved since the
    caller read the document (``base_revision`` mismatch), or the file
    changed on disk since it was loaded (manual edit).
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        *,
        readonly: bool = False,
        create: bool = True,
    ) -> None:
        self.path = pathlib.Path(path)
        self.readonly = readonly
        self.revision = 0
        self.lock = threading.RLock()
        self._digest: str | None = None
        self._document = new_document()
        if self.path.exists():
            self.reload()
        elif create and not readonly:
            self._write(self._document)
        else:
            raise FileNotFoundError(self.path)

    @property
    def document(self) -> AnnotationDocument:
        """Current document. Treat as read-only; edit through :meth:`apply`."""
        return self._document

    def snapshot(self) -> AnnotationDocument:
        """Return a deep copy of the current document."""
        with self.lock:
            return copy.deepcopy(self._document)

    def disk_changed(self) -> bool:
        """Return True if the file differs from what was last loaded/saved."""
        try:
            return _digest(self.path.read_bytes()) != self._digest
        except FileNotFoundError:
            return True

    def reload(self) -> None:
        """Re-read the file from disk.

        Raises:
            AnnotationError: If the file is invalid; the current document
                is kept in that case.

        """
        with self.lock:
            data = self.path.read_bytes()
            try:
                raw = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AnnotationError([f"{self.path}: invalid JSON ({exc})"]) from exc
            self._document = document_from_dict(raw)
            self._digest = _digest(data)
            self.revision += 1

    def _write(self, doc: AnnotationDocument) -> None:
        text = dump_annotations(doc).encode("utf-8")
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_bytes(text)
        if self.path.exists():
            bak = self.path.with_name(self.path.name + ".bak")
            bak.write_bytes(self.path.read_bytes())
        os.replace(tmp, self.path)
        self._digest = _digest(text)

    def apply(
        self,
        edit: Callable[[AnnotationDocument], None],
        *,
        base_revision: int | None = None,
        force: bool = False,
    ) -> int:
        """Apply ``edit`` to a copy of the document, validate it and save it.

        Args:
            edit: Function mutating the document copy in place.
            base_revision: Revision the caller's view was built from.
            force: Overwrite even if a conflict is detected.

        Returns:
            The new revision.

        Raises:
            ReadOnlyError: If the store is read-only.
            ConflictError: If the file changed elsewhere and ``force`` is False.
            AnnotationError: If the edited document is invalid.

        """
        if self.readonly:
            raise ReadOnlyError(f"{self.path} is opened read-only")
        with self.lock:
            if not force:
                if base_revision is not None and base_revision != self.revision:
                    raise ConflictError(
                        "The annotations were saved from another tab since "
                        "this editor was opened."
                    )
                if self.disk_changed():
                    raise ConflictError(
                        f"{self.path.name} was modified on disk since it was loaded."
                    )
            doc = copy.deepcopy(self._document)
            edit(doc)
            errors = validate(doc)
            if errors:
                raise AnnotationError(errors)
            self._write(doc)
            self._document = doc
            self.revision += 1
            return self.revision


# ── JSON Schema & template ──────────────────────────────────────────


def json_schema() -> dict[str, Any]:
    """Return a JSON Schema describing the annotation file format."""
    string = {"type": "string"}
    strings = {"type": "array", "items": string}
    figure = {
        "type": "object",
        "required": ["variable"],
        "properties": {
            "variable": string,
            "plots": {"type": "array", "items": {"enum": list(PLOT_LABELS)}},
            "expression": string,
            "zoom": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
            "caption": string,
        },
    }
    annotation = {
        "type": "object",
        "required": ["id"],
        "properties": {
            "id": string,
            "title": string,
            "category": string,
            "status": {"enum": list(STATUSES)},
            "variables": strings,
            "text": string,
            "effect": string,
            "magnitude": string,
            "cause": string,
            "refs": strings,
            "figures": {"type": "array", "items": figure},
        },
    }
    section = {
        "type": "object",
        "required": ["id"],
        "properties": {
            "id": string,
            "title": string,
            "body": string,
            "annotations": strings,
            "auto": {"enum": list(AUTO_SECTIONS)},
            "subsections": {"type": "array", "items": {"$ref": "#/$defs/section"}},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ncjuxta annotation document",
        "type": "object",
        "$defs": {"section": section},
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "language": {"enum": list(LANGUAGES)},
            "title": string,
            "subtitle": string,
            "date": string,
            "authors": strings,
            "source_label": string,
            "target_label": string,
            "datasets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"label": string, "value": string},
                },
            },
            "categories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {
                        "id": string,
                        "label": string,
                        "color": string,
                        "description": string,
                    },
                },
            },
            "sections": {"type": "array", "items": {"$ref": "#/$defs/section"}},
            "annotations": {"type": "array", "items": annotation},
        },
    }


def template_from_results(
    results: list[GroupComparisonResult], *, source: str = "", target: str = ""
) -> AnnotationDocument:
    """Build a starting document with one empty annotation per differing variable."""
    doc = new_document()
    doc.title = "Comparison report"
    if source:
        doc.datasets.append(DatasetRow("Source", source))
    if target:
        doc.datasets.append(DatasetRow("Target", target))
    ids: list[str] = []
    for group in results:
        for var in group.variables:
            if var.status not in DIFFERENT_STATUSES:
                continue
            ann_id = unique_id(var.varpath.strip("/").replace("/", "-"), ids)
            ids.append(ann_id)
            doc.annotations.append(
                Annotation(
                    id=ann_id,
                    title=var.long_name or var.varname,
                    variables=[var.varpath],
                    text="TODO",
                    figures=[FigureSpec(variable=var.varpath)],
                )
            )
    doc.sections = [
        Section("summary", "Summary", body="TODO", auto="summary_counts"),
        Section("findings", "Findings", annotations=ids),
        Section("agreement", "Agreement status", auto="agreement_table"),
    ]
    return doc


def _scan(
    source: str, target: str, tolerance_steps: float
) -> list[GroupComparisonResult]:
    import netCDF4

    from .comparison import scan_all_groups

    with (
        netCDF4.Dataset(source, mode="r") as src,
        netCDF4.Dataset(target, mode="r") as tgt,
    ):
        return scan_all_groups(src, tgt, None, tolerance_steps=tolerance_steps)


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``python -m app.annotations``."""
    parser = argparse.ArgumentParser(
        prog="python -m app.annotations",
        description="Utilities for ncjuxta annotation files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("schema", help="Print the JSON Schema of the file format.")
    p_tpl = sub.add_parser(
        "template", help="Write a skeleton with one annotation per difference."
    )
    p_tpl.add_argument("source")
    p_tpl.add_argument("target")
    p_tpl.add_argument("-o", "--output", default="-")
    p_tpl.add_argument("--tolerance-steps", type=float, default=DEFAULT_TOLERANCE_STEPS)
    p_chk = sub.add_parser("check", help="Validate a file (and its coverage).")
    p_chk.add_argument("file")
    p_chk.add_argument("source", nargs="?")
    p_chk.add_argument("target", nargs="?")
    p_chk.add_argument("--tolerance-steps", type=float, default=DEFAULT_TOLERANCE_STEPS)
    args = parser.parse_args(argv)

    if args.command == "schema":
        print(json.dumps(json_schema(), indent=2))
        return 0

    if args.command == "template":
        results = _scan(args.source, args.target, args.tolerance_steps)
        doc = template_from_results(
            results,
            source=pathlib.Path(args.source).name,
            target=pathlib.Path(args.target).name,
        )
        text = dump_annotations(doc)
        if args.output == "-":
            sys.stdout.write(text)
        else:
            pathlib.Path(args.output).write_text(text, encoding="utf-8")
        return 0

    try:
        doc = load_annotations(args.file)
    except AnnotationError as exc:
        for err in exc.errors:
            print(f"error: {err}", file=sys.stderr)
        return 1
    if args.source and args.target:
        warnings, unannotated = check_against_results(
            doc, _scan(args.source, args.target, args.tolerance_steps)
        )
        for msg in warnings:
            print(f"warning: {msg}", file=sys.stderr)
        for varpath in unannotated:
            print(f"unannotated: {varpath}", file=sys.stderr)
        if unannotated:
            return 2
    print(f"{args.file}: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
