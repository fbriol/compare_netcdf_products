"""Standalone annotated report combining analyst remarks and comparison plots.

The report is a single HTML file: annotations from an
:class:`~app.annotations.AnnotationDocument` are rendered as cards next to
the statistics computed by ncjuxta and the Plotly figures of the
annotated variables. The interactive variant embeds plotly.js and draws
figures lazily as they scroll into view. The PDF export prints that same
page with headless Chromium (optional ``playwright`` dependency).
"""

from __future__ import annotations

import dataclasses
import datetime
import html
import json
import logging
import math
import re
from typing import TYPE_CHECKING, Any

import numpy

from .annotations import DIFFERENT_STATUSES, match_annotations
from .constants import (
    DEFAULT_TOLERANCE_STEPS,
    PLOT_LABELS,
    STATUS_COLORS,
    STATUS_TEXT_COLORS,
)
from .data_loading import build_data_source
from .expression import create_expression_mask
from .plots import build_all_figures
from .utils import static_script

if TYPE_CHECKING:
    import netCDF4
    import plotly.graph_objects as go

    from .annotations import Annotation, AnnotationDocument, FigureSpec, Section
    from .comparison import GroupComparisonResult, VariableComparisonResult
    from .data_loading import DataSource
    from .flags import Flag

LOGGER = logging.getLogger(__name__)

#: Maximum number of variable rows listed inside one annotation card.
MAX_VARIABLE_ROWS = 50

_TEXTS: dict[str, dict[str, str]] = {
    "en": {
        "contents": "Contents",
        "filter": "Filter",
        "search": "Search variables or text…",
        "categories": "Categories",
        "statuses": "Status",
        "render_all": "Render all figures",
        "compared_data": "Compared data",
        "source_file": "Source file",
        "target_file": "Target file",
        "active_filter": "Global filter",
        "tolerance_rule": "Comparison tolerance",
        "tolerance_value": "{n} storage steps (scale_factor)",
        "generated": "Generated",
        "variables": "Variables",
        "magnitude": "Magnitude",
        "cause": "Cause",
        "refs": "References",
        "effect": "Effect:",
        "variable": "Variable",
        "status": "Status",
        "exceed": "Out of tolerance",
        "max_diff": "Max |diff|",
        "mean_diff": "Mean |diff|",
        "tolerance": "Tolerance",
        "annotations": "Annotations",
        "not_annotated": "not annotated",
        "more_variables": "… and {n} more variables",
        "other_annotations": "Other annotations",
        "status_counts": "Variables by status",
        "category_counts": "Annotations by category",
        "unannotated_count": "Differing variables without annotation",
        "count": "Count",
        "none": "None.",
        "uncategorised": "Uncategorised",
        "expression_error": "Cannot evaluate expression: {error}",
        "figure_error": "Cannot render figure for {variable}: {error}",
        "zoom": "zoom",
        "loading": "Figure rendered when scrolled into view.",
        "fullscreen": "Full screen (Esc to exit)",
        "open": "open",
        "confirmed": "confirmed",
        "resolved": "resolved",
        "wontfix": "won't fix",
        "equal": "equal",
        "almost_equal": "almost equal",
        "different": "different",
        "masked": "masked",
        "missing_in_source": "missing in source",
        "missing_in_target": "missing in target",
    },
    "fr": {
        "contents": "Sommaire",
        "filter": "Filtrer",
        "search": "Rechercher une variable ou un texte…",
        "categories": "Catégories",
        "statuses": "État",
        "render_all": "Afficher toutes les figures",
        "compared_data": "Données comparées",
        "source_file": "Fichier source",
        "target_file": "Fichier cible",
        "active_filter": "Filtre global",
        "tolerance_rule": "Tolérance de comparaison",
        "tolerance_value": "{n} pas de stockage (scale_factor)",
        "generated": "Généré le",
        "variables": "Variables",
        "magnitude": "Ampleur",
        "cause": "Cause",
        "refs": "Références",
        "effect": "Effet :",
        "variable": "Variable",
        "status": "Statut",
        "exceed": "Hors tolérance",
        "max_diff": "|écart| max",
        "mean_diff": "|écart| moyen",
        "tolerance": "Tolérance",
        "annotations": "Annotations",
        "not_annotated": "non annoté",
        "more_variables": "… et {n} autres variables",
        "other_annotations": "Autres annotations",
        "status_counts": "Variables par statut",
        "category_counts": "Annotations par catégorie",
        "unannotated_count": "Variables en écart sans annotation",
        "count": "Nombre",
        "none": "Aucune.",
        "uncategorised": "Sans catégorie",
        "expression_error": "Expression invalide : {error}",
        "figure_error": "Impossible de tracer {variable} : {error}",
        "zoom": "zoom",
        "loading": "Figure affichée au défilement.",
        "fullscreen": "Plein écran (Échap pour quitter)",
        "open": "ouvert",
        "confirmed": "confirmé",
        "resolved": "résolu",
        "wontfix": "sans suite",
        "equal": "identique",
        "almost_equal": "quasi identique",
        "different": "différent",
        "masked": "masqué",
        "missing_in_source": "absent de la source",
        "missing_in_target": "absent de la cible",
    },
}

_STATUS_ORDER = (
    "different",
    "almost_equal",
    "missing_in_source",
    "missing_in_target",
    "masked",
    "equal",
)


@dataclasses.dataclass
class ReportInputs:
    """Data the report is built from.

    Attributes:
        src_ds: Open source dataset.
        tgt_ds: Open target dataset.
        src_name: Display name of the source file.
        tgt_name: Display name of the target file.
        results: Comparison results (``scan_all_groups``).
        flags: Flag variables passed to the data sources.
        mask: Global filter mask applied to figures without expression.
        expression: Text of the global filter, shown in the header.
        tolerance_steps: Tolerance used for the results, in storage steps.

    """

    src_ds: netCDF4.Dataset
    tgt_ds: netCDF4.Dataset
    src_name: str
    tgt_name: str
    results: list[GroupComparisonResult]
    flags: list[Flag] | None = None
    mask: numpy.ndarray | None = None
    expression: str | None = None
    tolerance_steps: float = DEFAULT_TOLERANCE_STEPS


# ── Figure helpers ──────────────────────────────────────────────────


def figure_to_json(fig: go.Figure) -> str:
    """Serialise a figure for embedding in a ``<script>`` element.

    Numeric arrays keep plotly's compact base64 typed-array encoding
    (decoded by the plotly.js bundled with plotly.py), NaN become
    ``null`` and ``<`` is escaped.
    """
    from plotly.utils import PlotlyJSONEncoder

    text = json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder)
    return text.replace("<", "\\u003c")


def relabel_figure(fig: go.Figure, source_label: str, target_label: str) -> None:
    """Replace the words *Source*/*Target* in trace names and axis titles."""
    if (source_label, target_label) == ("Source", "Target"):
        return
    names = {"Source": source_label, "Target": target_label}
    pattern = re.compile(r"\b(Source|Target)\b")

    def _sub(text: str) -> str:
        return pattern.sub(lambda m: names[m.group(1)], text)

    for trace in fig.data:
        name = getattr(trace, "name", None)
        if isinstance(name, str) and name:
            trace.name = _sub(name)
    for axis in ("xaxis", "yaxis"):
        layout_axis = getattr(fig.layout, axis, None)
        title = getattr(getattr(layout_axis, "title", None), "text", None)
        if layout_axis is not None and isinstance(title, str):
            layout_axis.title.text = _sub(title)


def _restrict_to_zoom(data_source: DataSource, zoom: list[float]) -> None:
    """Keep only the indices within ``zoom`` (statistics included)."""
    n = len(data_source.full_x)
    lo = max(0, math.ceil(zoom[0]))
    hi = min(n - 1, math.floor(zoom[1]))
    if hi < lo:
        return
    data_source.apply_flag_filter(numpy.arange(lo, hi + 1))


@dataclasses.dataclass
class _RenderedFigure:
    label: str
    payload: str


class _FigureFactory:
    """Build figures for figure specs, caching masks and data sources."""

    def __init__(self, inputs: ReportInputs, doc: AnnotationDocument) -> None:
        self.inputs = inputs
        self.doc = doc
        self._masks: dict[str, tuple[numpy.ndarray | None, str | None]] = {}
        self._long_names = {
            v.varpath: v.long_name or v.varname
            for group in inputs.results
            for v in group.variables
        }

    def _mask(self, expression: str) -> tuple[numpy.ndarray | None, str | None]:
        if not expression.strip():
            return self.inputs.mask, None
        if expression not in self._masks:
            self._masks[expression] = create_expression_mask(
                self.inputs.src_ds, expression
            )
        return self._masks[expression]

    def render(self, spec: FigureSpec) -> tuple[list[_RenderedFigure], str | None]:
        """Return the rendered figures of a spec, or an error message."""
        mask, error = self._mask(spec.expression)
        if error is not None:
            return [], error
        data_source = build_data_source(
            self.inputs.src_ds,
            self.inputs.tgt_ds,
            spec.variable,
            mask,
            self.inputs.flags,
        )
        if spec.zoom is not None:
            _restrict_to_zoom(data_source, spec.zoom)
        long_name = self._long_names.get(spec.variable, spec.variable)
        wanted = set(spec.plots) if spec.plots else set(PLOT_LABELS)
        out: list[_RenderedFigure] = []
        for label, fig in build_all_figures(long_name, data_source):
            if label not in wanted:
                continue
            relabel_figure(fig, self.doc.source_label, self.doc.target_label)
            if spec.zoom is not None and label in {
                "Signal Comparison",
                "Signal Difference",
                "Exclusive Undefined Positions",
            }:
                fig.update_xaxes(range=list(spec.zoom))
            out.append(_RenderedFigure(label, figure_to_json(fig)))
        return out, None


# ── HTML helpers ────────────────────────────────────────────────────


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _markdown_renderer() -> Any:
    from markdown_it import MarkdownIt

    return MarkdownIt("commonmark", {"html": False, "typographer": True}).enable(
        ["table", "strikethrough"]
    )


def _fmt(value: object, language: str = "en") -> str:
    """Format a number for display, with French separators when asked."""
    if value is None:
        return "—"
    if isinstance(value, (float, numpy.floating)):
        if math.isnan(float(value)):
            return "—"
        text = f"{float(value):.6g}"
        return text.replace(".", ",") if language == "fr" else text
    if isinstance(value, (int, numpy.integer)):
        text = f"{int(value):,}"
        return text.replace(",", "\u202f") if language == "fr" else text
    return str(value)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-") or "x"


class _Builder:
    def __init__(
        self,
        doc: AnnotationDocument,
        inputs: ReportInputs,
        *,
        generated_at: datetime.datetime | None,
    ) -> None:
        self.doc = doc
        self.inputs = inputs
        self.generated_at = generated_at or datetime.datetime.now()
        self.t = _TEXTS.get(doc.language, _TEXTS["en"])
        self.md = _markdown_renderer()
        self.figures = _FigureFactory(inputs, doc)
        self.variables: dict[str, VariableComparisonResult] = {
            v.varpath: v for group in inputs.results for v in group.variables
        }
        self.coverage = match_annotations(doc, self.variables)
        self.anchors = {
            a.id: f"ann-{i}-{_slug(a.id)}" for i, a in enumerate(doc.annotations)
        }
        self.section_numbers: dict[str, str] = {}
        self.toc: list[tuple[int, str, str]] = []
        self.figure_payloads: list[str] = []
        self.has_plotly = False

    # -- small pieces --------------------------------------------------

    def fmt(self, value: object) -> str:
        return _fmt(value, self.doc.language)

    def markdown(self, text: str) -> str:
        return self.md.render(text) if text.strip() else ""

    def status_chip(self, status: str) -> str:
        bg = STATUS_COLORS.get(status, "#e0e0e0")
        fg = STATUS_TEXT_COLORS.get(status, "#000")
        return (
            f'<span class="vstatus" style="background:{_esc(bg)};color:{_esc(fg)}">'
            f"{_esc(self.t.get(status, status))}</span>"
        )

    def category_badge(self, category_id: str) -> str:
        cat = self.doc.category(category_id)
        label = (cat.label or cat.id) if cat else self.t["uncategorised"]
        color = cat.color if cat else "#9e9e9e"
        return f'<span class="badge" style="--c:{_esc(color)}">{_esc(label)}</span>'

    def annotation_link(self, ann: Annotation) -> str:
        cat = self.doc.category(ann.category)
        color = cat.color if cat else "#9e9e9e"
        return (
            f'<a class="annlink" href="#{self.anchors[ann.id]}" '
            f'style="--c:{_esc(color)}">{_esc(ann.title or ann.id)}</a>'
        )

    # -- variables -----------------------------------------------------

    def variable_row(self, varpath: str, with_annotations: bool) -> str:
        var = self.variables[varpath]
        stats = var.stats
        if stats is not None and stats.n_valid:
            pct = 100.0 * stats.n_exceed / stats.n_valid
            exceed = (
                f"{self.fmt(stats.n_exceed)} / {self.fmt(stats.n_valid)} "
                f"({self.fmt(round(pct, 1))} %)"
            )
        else:
            exceed = "—"
        cells = [
            f'<td><code>{_esc(varpath)}</code><div class="ln">'
            f"{_esc(var.long_name)}</div></td>",
            f"<td>{self.status_chip(var.status)}</td>",
            f'<td class="num">{_esc(exceed)}</td>',
            f'<td class="num">{_esc(self.fmt(stats.max_diff if stats else None))}</td>',
            f'<td class="num">{_esc(self.fmt(stats.mean_diff if stats else None))}</td>',
            f'<td class="num">{_esc(self.fmt(var.atol or None))}</td>',
        ]
        classes = ["var-row"]
        if with_annotations:
            anns = self.coverage.get(varpath, [])
            if anns:
                links = " ".join(self.annotation_link(a) for a in anns)
            else:
                links = (
                    f'<span class="unannotated">{_esc(self.t["not_annotated"])}</span>'
                )
                classes.append("is-unannotated")
            cells.append(f"<td>{links}</td>")
        search = f"{varpath} {var.long_name}".lower()
        return (
            f'<tr class="{" ".join(classes)}" data-search="{_esc(search)}">'
            + "".join(cells)
            + "</tr>"
        )

    def variable_table(self, varpaths: list[str], *, with_annotations: bool) -> str:
        t = self.t
        head = [
            t["variable"],
            t["status"],
            t["exceed"],
            t["max_diff"],
            t["mean_diff"],
            t["tolerance"],
        ]
        if with_annotations:
            head.append(t["annotations"])
        rows = [self.variable_row(v, with_annotations) for v in varpaths]
        return (
            '<div class="table-wrap"><table class="vars sortable"><thead><tr>'
            + "".join(f"<th>{_esc(h)}</th>" for h in head)
            + "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )

    # -- figures -------------------------------------------------------

    def figure_block(self, spec: FigureSpec) -> str:
        meta = [f"<code>{_esc(spec.variable)}</code>"]
        if spec.expression.strip():
            meta.append(f"<code>{_esc(spec.expression)}</code>")
        if spec.zoom is not None:
            meta.append(
                f"{_esc(self.t['zoom'])} [{self.fmt(spec.zoom[0])} ; {self.fmt(spec.zoom[1])}]"
            )
        parts = [
            f'<figure class="figblock"><div class="figmeta">{" · ".join(meta)}</div>'
        ]
        try:
            rendered, error = self.figures.render(spec)
        except Exception as exc:
            LOGGER.exception("Figure rendering failed for %s", spec.variable)
            rendered, error = [], None
            parts.append(
                '<p class="error">'
                + _esc(self.t["figure_error"].format(variable=spec.variable, error=exc))
                + "</p>"
            )
        if error is not None:
            parts.append(
                '<p class="error">'
                + _esc(self.t["expression_error"].format(error=error))
                + "</p>"
            )
        for fig in rendered:
            parts.append(
                '<div class="plotwrap ncj-plot-container"><div class="plothead">'
                f"<h5>{_esc(fig.label)}</h5>"
                '<button type="button" class="ncj-fullscreen-btn" '
                f'title="{_esc(self.t["fullscreen"])}">\u26f6</button></div>'
            )
            idx = len(self.figure_payloads)
            self.figure_payloads.append(fig.payload)
            self.has_plotly = True
            parts.append(
                f'<div class="plot" data-fig="{idx}">'
                f'<span class="placeholder">{_esc(self.t["loading"])}</span></div>'
            )
            parts.append("</div>")
        if spec.caption.strip():
            parts.append(f"<figcaption>{self.markdown(spec.caption)}</figcaption>")
        parts.append("</figure>")
        return "".join(parts)

    # -- annotations ---------------------------------------------------

    def annotation_card(self, ann: Annotation) -> str:
        t = self.t
        matched = [
            v for v in self.variables if any(a is ann for a in self.coverage.get(v, []))
        ]
        search = " ".join(
            [
                ann.id,
                ann.title,
                ann.text,
                ann.cause,
                ann.magnitude,
                *ann.variables,
                *matched,
            ]
        ).lower()
        facts = []
        if ann.variables:
            facts.append(
                (
                    t["variables"],
                    " ".join(f"<code>{_esc(v)}</code>" for v in ann.variables),
                )
            )
        if ann.magnitude:
            facts.append((t["magnitude"], _esc(ann.magnitude)))
        if ann.cause:
            facts.append((t["cause"], _esc(ann.cause)))
        if ann.refs:
            facts.append((t["refs"], ", ".join(_esc(r) for r in ann.refs)))
        body = []
        if facts:
            body.append(
                '<dl class="facts">'
                + "".join(f"<dt>{_esc(k)}</dt><dd>{v}</dd>" for k, v in facts)
                + "</dl>"
            )
        body.append(self.markdown(ann.text))
        if ann.effect.strip():
            body.append(
                f'<div class="effect"><strong>{_esc(t["effect"])}</strong>'
                + self.markdown(ann.effect)
                + "</div>"
            )
        if matched:
            shown = matched[:MAX_VARIABLE_ROWS]
            body.append(self.variable_table(shown, with_annotations=False))
            if len(matched) > MAX_VARIABLE_ROWS:
                body.append(
                    '<p class="more">'
                    + _esc(t["more_variables"].format(n=len(matched) - len(shown)))
                    + "</p>"
                )
        for spec in ann.figures:
            body.append(self.figure_block(spec))
        return (
            f'<details class="annotation" open id="{self.anchors[ann.id]}" '
            f'data-category="{_esc(ann.category)}" data-status="{_esc(ann.status)}" '
            f'data-search="{_esc(search)}">'
            f"<summary>{self.category_badge(ann.category)}"
            f'<span class="astatus s-{_esc(ann.status)}">{_esc(t.get(ann.status, ann.status))}</span>'
            f'<span class="atitle">{_esc(ann.title or ann.id)}</span>'
            f'<a class="anchor" href="#{self.anchors[ann.id]}" title="{_esc(ann.id)}">#</a>'
            f'</summary><div class="abody">{"".join(body)}</div></details>'
        )

    # -- auto sections -------------------------------------------------

    def differing_varpaths(self) -> list[str]:
        rank = {s: i for i, s in enumerate(_STATUS_ORDER)}
        paths = [p for p, v in self.variables.items() if v.status in DIFFERENT_STATUSES]
        return sorted(paths, key=lambda p: (rank.get(self.variables[p].status, 99), p))

    def auto_section(self, kind: str) -> str:
        t = self.t
        if kind == "agreement_table":
            paths = self.differing_varpaths()
            if not paths:
                return f"<p>{_esc(t['none'])}</p>"
            return self.variable_table(paths, with_annotations=True)
        if kind == "unannotated":
            paths = [p for p in self.differing_varpaths() if p not in self.coverage]
            if not paths:
                return f"<p>{_esc(t['none'])}</p>"
            return self.variable_table(paths, with_annotations=False)
        counts: dict[str, int] = {}
        for var in self.variables.values():
            counts[var.status] = counts.get(var.status, 0) + 1
        status_rows = "".join(
            f"<tr><td>{self.status_chip(s)}</td><td class='num'>{self.fmt(counts[s])}</td></tr>"
            for s in _STATUS_ORDER
            if s in counts
        )
        cat_counts: dict[str, int] = {}
        for ann in self.doc.annotations:
            cat_counts[ann.category] = cat_counts.get(ann.category, 0) + 1
        cat_rows = "".join(
            f"<tr><td>{self.category_badge(c.id)}</td>"
            f"<td class='num'>{cat_counts.get(c.id, 0)}</td></tr>"
            for c in self.doc.categories
        )
        if cat_counts.get(""):
            cat_rows += (
                f"<tr><td>{self.category_badge('')}</td>"
                f"<td class='num'>{cat_counts['']}</td></tr>"
            )
        unannotated = sum(
            1 for p in self.differing_varpaths() if p not in self.coverage
        )
        return (
            '<div class="counts">'
            f"<table class='compact'><caption>{_esc(t['status_counts'])}</caption>"
            f"<tbody>{status_rows}</tbody></table>"
            f"<table class='compact'><caption>{_esc(t['category_counts'])}</caption>"
            f"<tbody>{cat_rows}</tbody></table>"
            f"<table class='compact'><caption>{_esc(t['unannotated_count'])}</caption>"
            f"<tbody><tr><td>{_esc(t['count'])}</td><td class='num'>{unannotated}</td>"
            "</tr></tbody></table></div>"
        )

    # -- sections ------------------------------------------------------

    def section(self, section: Section, number: str, depth: int) -> str:
        anchor = f"sec-{_slug(section.id)}"
        self.section_numbers[section.id] = number
        self.toc.append((depth, f"{number}. {section.title}", anchor))
        level = min(2 + depth, 6)
        parts = [
            f'<section class="sec depth-{depth}" id="{anchor}">'
            f"<h{level}>{_esc(number)}. {_esc(section.title)}</h{level}>",
            self.markdown(section.body),
        ]
        if section.auto:
            parts.append(self.auto_section(section.auto))
        for ann_id in section.annotations:
            ann = self.doc.annotation(ann_id)
            if ann is not None:
                parts.append(self.annotation_card(ann))
        for i, sub in enumerate(section.subsections, start=1):
            parts.append(self.section(sub, f"{number}.{i}", depth + 1))
        parts.append("</section>")
        return "".join(parts)

    # -- page ----------------------------------------------------------

    def header(self) -> str:
        doc, t = self.doc, self.t
        byline = " · ".join(_esc(x) for x in [doc.date, ", ".join(doc.authors)] if x)
        rows = [(r.label, r.value) for r in doc.datasets]
        rows += [
            (f"{t['source_file']} ({doc.source_label})", self.inputs.src_name),
            (f"{t['target_file']} ({doc.target_label})", self.inputs.tgt_name),
        ]
        if self.inputs.expression:
            rows.append((t["active_filter"], self.inputs.expression))
        rows.append(
            (
                t["tolerance_rule"],
                t["tolerance_value"].format(n=self.fmt(self.inputs.tolerance_steps)),
            )
        )
        rows.append((t["generated"], self.generated_at.strftime("%Y-%m-%d %H:%M")))
        table = "".join(
            f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in rows
        )
        return (
            "<header class='doc-header'>"
            f"<h1>{_esc(doc.title or 'Comparison report')}</h1>"
            + (f"<p class='subtitle'>{_esc(doc.subtitle)}</p>" if doc.subtitle else "")
            + (f"<p class='byline'>{byline}</p>" if byline else "")
            + f"<h2 class='plain'>{_esc(t['compared_data'])}</h2>"
            f"<table class='datasets'><tbody>{table}</tbody></table></header>"
        )

    def sidebar(self) -> str:
        t = self.t
        toc = "".join(
            f'<li class="d{min(d, 3)}"><a href="#{a}">{_esc(label)}</a></li>'
            for d, label, a in self.toc
        )
        used_cats = {a.category for a in self.doc.annotations}
        cats = "".join(
            f'<button type="button" class="chip on" data-kind="category" '
            f'data-value="{_esc(c.id)}" style="--c:{_esc(c.color)}">'
            f"{_esc(c.label or c.id)}</button>"
            for c in self.doc.categories
            if c.id in used_cats
        )
        if "" in used_cats:
            cats += (
                '<button type="button" class="chip on" data-kind="category" '
                f'data-value="" style="--c:#9e9e9e">{_esc(t["uncategorised"])}</button>'
            )
        statuses = sorted({a.status for a in self.doc.annotations})
        stats = "".join(
            f'<button type="button" class="chip on" data-kind="status" '
            f'data-value="{_esc(s)}">{_esc(t.get(s, s))}</button>'
            for s in statuses
        )
        filters = ""
        if self.doc.annotations:
            filters = (
                f"<h3>{_esc(t['filter'])}</h3>"
                f'<input id="q" type="search" placeholder="{_esc(t["search"])}">'
                + (
                    f"<h4>{_esc(t['categories'])}</h4><div class='chips'>{cats}</div>"
                    if cats
                    else ""
                )
                + (
                    f"<h4>{_esc(t['statuses'])}</h4><div class='chips'>{stats}</div>"
                    if stats
                    else ""
                )
            )
        render_all = (
            f'<button type="button" id="render-all">{_esc(t["render_all"])}</button>'
            if self.has_plotly
            else ""
        )
        return (
            '<nav id="side"><div class="side-inner">'
            f"<h3>{_esc(t['contents'])}</h3><ol class='toc'>{toc}</ol>"
            f"{filters}{render_all}</div></nav>"
        )

    def build(self, plotlyjs: str) -> str:
        doc = self.doc
        body = [self.header()]
        for i, section in enumerate(doc.sections, start=1):
            body.append(self.section(section, str(i), 0))
        unplaced = doc.unplaced_annotations()
        if unplaced:
            number = str(len(doc.sections) + 1)
            anchor = "sec-other-annotations"
            self.toc.append((0, f"{number}. {self.t['other_annotations']}", anchor))
            body.append(
                f'<section class="sec depth-0" id="{anchor}">'
                f"<h2>{number}. {_esc(self.t['other_annotations'])}</h2>"
                + "".join(self.annotation_card(a) for a in unplaced)
                + "</section>"
            )
        head_script = ""
        if self.has_plotly:
            head_script = _plotly_script(plotlyjs)
        payloads = "".join(
            f'<script type="application/json" id="fig-{i}">{p}</script>'
            for i, p in enumerate(self.figure_payloads)
        )
        return (
            "<!DOCTYPE html>"
            f'<html lang="{_esc(doc.language)}"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{_esc(doc.title or 'Comparison report')}</title>"
            f"<style>{_CSS}</style>{head_script}</head><body>"
            f"{self.sidebar()}<main>{''.join(body)}</main>"
            f"{payloads}<script>{_JS}</script>"
            f"<script>{static_script('fullscreen.js')}</script></body></html>"
        )


def _plotly_script(mode: str) -> str:
    if mode == "cdn":
        from plotly.offline import get_plotlyjs_version

        return (
            '<script src="https://cdn.plot.ly/plotly-'
            f'{get_plotlyjs_version()}.min.js" charset="utf-8"></script>'
        )
    from plotly.offline import get_plotlyjs

    return f"<script>{get_plotlyjs()}</script>"


def build_report(
    doc: AnnotationDocument,
    inputs: ReportInputs,
    *,
    plotlyjs: str = "inline",
    generated_at: datetime.datetime | None = None,
) -> str:
    """Render the annotated report as a standalone HTML page.

    Args:
        doc: The annotation document.
        inputs: Datasets and comparison results.
        plotlyjs: ``"inline"`` embeds plotly.js (works offline),
            ``"cdn"`` references it from cdn.plot.ly.
        generated_at: Timestamp shown in the header (defaults to now).

    Returns:
        The HTML document.

    """
    if plotlyjs not in {"inline", "cdn"}:
        raise ValueError(f"plotlyjs must be 'inline' or 'cdn', not {plotlyjs!r}")
    builder = _Builder(doc, inputs, generated_at=generated_at)
    return builder.build(plotlyjs)


def html_to_pdf(page: str, *, timeout_ms: int = 120_000) -> bytes:
    """Print a report page to A4 PDF with headless Chromium.

    Every figure is drawn before printing. Requires the optional ``pdf``
    extra (``pip install 'ncjuxta[pdf]'`` then ``playwright install
    chromium``).

    Raises:
        RuntimeError: If Playwright or its Chromium build is missing.

    """
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "PDF export requires Playwright: pip install 'ncjuxta[pdf]' "
            "&& playwright install chromium"
        ) from exc
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "report.html"
        path.write_text(page, encoding="utf-8")
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                try:
                    tab = browser.new_page(viewport={"width": 690, "height": 1000})
                    tab.goto(path.as_uri(), timeout=timeout_ms)
                    tab.emulate_media(media="print")
                    tab.evaluate("window.ncjuxtaRenderAll && window.ncjuxtaRenderAll()")
                    tab.wait_for_function(
                        "!document.querySelector('.plot .placeholder')",
                        timeout=timeout_ms,
                    )
                    pdf: bytes = tab.pdf(
                        format="A4",
                        print_background=True,
                        margin={
                            "top": "16mm",
                            "bottom": "16mm",
                            "left": "14mm",
                            "right": "14mm",
                        },
                        display_header_footer=True,
                        header_template="<span></span>",
                        footer_template=(
                            '<div style="font-size:8px;width:100%;text-align:center;'
                            'color:#777"><span class="pageNumber"></span> / '
                            '<span class="totalPages"></span></div>'
                        ),
                    )
                finally:
                    browser.close()
        except PlaywrightError as exc:
            raise RuntimeError(
                f"PDF export failed ({exc}). Run 'playwright install chromium'."
            ) from exc
    return pdf


def build_pdf(doc: AnnotationDocument, inputs: ReportInputs) -> bytes:
    """Render the annotated report as PDF (see :func:`html_to_pdf`)."""
    return html_to_pdf(build_report(doc, inputs))


_CSS = """
:root{--fg:#1f2933;--muted:#5f6b7a;--line:#d9dee5;--bg:#ffffff;--panel:#f5f7fa;
--accent:rgb(76,114,176);--code:#8a5a00}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 "Segoe UI",system-ui,-apple-system,sans-serif;
color:var(--fg);background:var(--bg);display:flex;align-items:flex-start}
#side{position:sticky;top:0;height:100vh;width:280px;flex:0 0 280px;overflow:auto;
background:var(--panel);border-right:1px solid var(--line);font-size:13px}
.side-inner{padding:16px}
#side h3{margin:12px 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;
color:var(--muted)}
#side h4{margin:10px 0 4px;font-size:12px;color:var(--muted)}
.toc{list-style:none;padding:0;margin:0}
.toc li{margin:2px 0}.toc .d1{padding-left:12px}.toc .d2,.toc .d3{padding-left:24px}
.toc a{color:var(--fg);text-decoration:none}.toc a:hover{color:var(--accent)}
#q{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:4px;font:inherit}
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{border:1px solid var(--c,#90a4ae);background:#fff;color:var(--fg);border-radius:12px;
padding:2px 9px;font:inherit;font-size:12px;cursor:pointer;opacity:.45}
.chip.on{opacity:1;box-shadow:inset 3px 0 0 var(--c,#90a4ae)}
#render-all{margin-top:16px;width:100%;padding:6px;border:1px solid var(--accent);
background:#fff;color:var(--accent);border-radius:4px;cursor:pointer;font:inherit}
main{flex:1;min-width:0;max-width:1100px;padding:24px 40px 80px}
h1{font-size:28px;font-weight:500;margin:8px 0 4px;line-height:1.25}
h2{font-size:22px;font-weight:500;margin:36px 0 10px;border-bottom:1px solid var(--line);
padding-bottom:4px}
h2.plain{font-size:16px;border:0;margin:20px 0 6px}
h3{font-size:18px;font-weight:500;margin:26px 0 8px}
h4,h5,h6{font-size:15px;margin:20px 0 6px}
.subtitle{font-size:17px;color:var(--muted);margin:0}.byline{color:var(--muted);margin:4px 0}
code{font-family:"SFMono-Regular",Consolas,monospace;font-size:.88em;color:var(--code)}
table{border-collapse:collapse;margin:10px 0;font-size:13.5px}
th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}
thead th{border-bottom:2px solid var(--fg);font-weight:600;cursor:default}
table.sortable thead th{cursor:pointer}
table.datasets th{font-weight:500;color:var(--muted);width:200px}
td.num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
.table-wrap{overflow-x:auto}
.ln{color:var(--muted);font-size:12px}
.badge{display:inline-block;border-radius:3px;padding:1px 8px;font-size:12px;
font-weight:600;color:#fff;background:var(--c)}
.vstatus{display:inline-block;border-radius:3px;padding:1px 6px;font-size:12px;
white-space:nowrap;border:1px solid rgba(0,0,0,.12)}
.astatus{font-size:12px;color:var(--muted);border:1px solid var(--line);border-radius:3px;
padding:0 6px}
.s-resolved{color:#2e7d32;border-color:#a5d6a7}.s-open{color:#c62828;border-color:#ef9a9a}
.annotation{border:1px solid var(--line);border-radius:6px;margin:14px 0;background:#fff}
.annotation>summary{display:flex;gap:10px;align-items:center;padding:10px 14px;cursor:pointer;
background:var(--panel);border-radius:6px 6px 0 0;list-style:none}
.annotation>summary::-webkit-details-marker{display:none}
.atitle{font-weight:600;flex:1}
.anchor{color:var(--muted);text-decoration:none;visibility:hidden}
.annotation>summary:hover .anchor{visibility:visible}
.abody{padding:4px 16px 12px}
.facts{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:10px 0;
font-size:14px}
.facts dt{color:var(--muted)}.facts dd{margin:0}
.effect{border-left:3px solid var(--accent);padding:2px 12px;background:var(--panel);
margin:10px 0}.effect p{margin:4px 0}
.annlink{display:inline-block;margin:1px 4px 1px 0;padding:0 6px;border-left:4px solid var(--c);
background:var(--panel);font-size:12px;color:var(--fg);text-decoration:none}
.unannotated{color:#c62828;font-size:12px;font-style:italic}
tr.is-unannotated td:first-child{box-shadow:inset 3px 0 0 #c62828}
.figblock{margin:16px 0;border-top:1px dashed var(--line);padding-top:8px}
.figmeta{font-size:12px;color:var(--muted)}
.plothead{display:flex;align-items:center;justify-content:space-between;margin:10px 0 2px}
.plotwrap h5{margin:0;font-size:13px;color:var(--muted);font-weight:500}
.ncj-fullscreen-btn{border:1px solid var(--line);border-radius:4px;background:#fff;
color:var(--muted);cursor:pointer;font-size:14px;line-height:1;padding:2px 7px}
.ncj-fullscreen-btn:hover{color:var(--accent);border-color:var(--accent)}
.ncj-plot-container:fullscreen{background:#fff;padding:12px 16px;overflow:auto}
.plot{min-height:320px;position:relative}
.placeholder{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
color:var(--muted);font-size:12px;background:var(--panel)}
figcaption{font-size:13px;color:var(--muted)}
.error{color:#c62828}.more{color:var(--muted);font-size:13px}
.counts{display:flex;flex-wrap:wrap;gap:24px}
table.compact caption{text-align:left;font-weight:600;padding-bottom:4px;white-space:nowrap}
.hidden{display:none!important}
@media (max-width:900px){body{display:block}#side{position:static;width:auto;height:auto}
main{padding:16px}}
@media print{#side{display:none}body{display:block}main{max-width:none;padding:0}
.anchor,.placeholder,.ncj-fullscreen-btn{display:none}h2,h3,h4,h5,.annotation>summary{break-after:avoid}
.plotwrap,tr,.facts,.effect{break-inside:avoid}.table-wrap{overflow:visible}
table.vars{width:100%;font-size:11px}table.vars td,table.vars th{padding:4px 6px}
table.vars code{word-break:break-all}}
@page{size:A4;margin:16mm 14mm}
"""

_JS = r"""
(function(){
  var plots=[].slice.call(document.querySelectorAll('.plot[data-fig]'));
  var live=new Set();
  var keepAll=false,printing=false;
  function printHeight(h){return Math.min(h||450,330);}
  function draw(el){
    if(live.has(el)||!window.Plotly)return;
    var fig=JSON.parse(document.getElementById('fig-'+el.dataset.fig).textContent);
    var ph=el.querySelector('.placeholder');if(ph)ph.remove();
    var layout=fig.layout||{};layout.autosize=true;delete layout.width;
    if(printing)layout.height=printHeight(layout.height);
    Plotly.newPlot(el,fig.data||[],layout,{responsive:true,displaylogo:false});
    live.add(el);
  }
  function drop(el){
    if(keepAll||!live.has(el))return;
    Plotly.purge(el);live.delete(el);
    var ph=document.createElement('span');ph.className='placeholder';
    ph.textContent='…';el.appendChild(ph);
  }
  if('IntersectionObserver' in window){
    var near=new IntersectionObserver(function(es){es.forEach(function(e){
      if(e.isIntersecting)draw(e.target);});},{rootMargin:'400px 0px'});
    var far=new IntersectionObserver(function(es){es.forEach(function(e){
      if(!e.isIntersecting&&live.size>12)drop(e.target);});},{rootMargin:'3000px 0px'});
    plots.forEach(function(el){near.observe(el);far.observe(el);});
  }else{plots.forEach(draw);}
  function renderAll(){
    keepAll=true;
    document.querySelectorAll('details.annotation').forEach(function(d){d.open=true;});
    plots.forEach(draw);
  }
  function renderForPrint(){
    if(printing)return;
    printing=true;
    live.forEach(function(el){
      Plotly.relayout(el,{height:printHeight(el.layout&&el.layout.height)});
    });
    renderAll();
  }
  window.ncjuxtaRenderAll=renderForPrint;
  var all=document.getElementById('render-all');
  if(all)all.addEventListener('click',renderAll);
  window.addEventListener('beforeprint',renderForPrint);

  var q=document.getElementById('q');
  var chips=[].slice.call(document.querySelectorAll('.chip'));
  function active(kind){
    var on=chips.filter(function(c){return c.dataset.kind===kind;});
    if(!on.length)return null;
    return new Set(on.filter(function(c){return c.classList.contains('on');})
      .map(function(c){return c.dataset.value;}));
  }
  function apply(){
    var text=(q&&q.value||'').toLowerCase().trim();
    var cats=active('category'),sts=active('status');
    document.querySelectorAll('details.annotation').forEach(function(d){
      var ok=(!cats||cats.has(d.dataset.category))&&(!sts||sts.has(d.dataset.status))
        &&(!text||d.dataset.search.indexOf(text)>=0);
      d.classList.toggle('hidden',!ok);
    });
    document.querySelectorAll('tr.var-row').forEach(function(r){
      r.classList.toggle('hidden',!!text&&r.dataset.search.indexOf(text)<0);
    });
  }
  if(q)q.addEventListener('input',apply);
  chips.forEach(function(c){c.addEventListener('click',function(){
    c.classList.toggle('on');apply();});});

  document.querySelectorAll('table.sortable').forEach(function(t){
    t.querySelectorAll('thead th').forEach(function(th,i){
      var asc=true;
      th.addEventListener('click',function(){
        var body=t.tBodies[0];var rows=[].slice.call(body.rows);
        function key(r){var s=r.cells[i].textContent.trim();
          var n=parseFloat(s.replace(/[,\s]/g,''));return isNaN(n)?s.toLowerCase():n;}
        rows.sort(function(a,b){var x=key(a),y=key(b);
          var c=(typeof x===typeof y)?(x<y?-1:x>y?1:0):(typeof x==='number'?-1:1);
          return asc?c:-c;});
        asc=!asc;rows.forEach(function(r){body.appendChild(r);});
      });
    });
  });
})();
"""
