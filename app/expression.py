"""Safe expression evaluation for filtering NetCDF data.

Expression syntax uses ``${/path/to/variable}`` to reference NetCDF
variables.  The evaluator is :mod:`numexpr` — safe by construction
(no attribute access, no Python-level calls, no imports) and 2-5x
faster than Python-level evaluation on large arrays.

Supported operators and functions follow numexpr's grammar:
arithmetic (``+ - * / ** %``), comparisons (``== != < <= > >=``),
bitwise logic (``& | ~``) on boolean arrays, and numexpr builtins
(``where``, ``sin``, ``cos``, ``exp``, ``log``, ``sqrt``, ``abs``,
``sum``).  In addition, ``np.isin(x, [a, b, ...])`` is translated to
``((x == a) | (x == b) | ...)`` automatically so existing filter
expressions keep working.

Examples::

    ${/data_20/distance_to_coast} > 30000
    (${/data_20/distance_to_coast} > 30000) & (${/data_20/flag} == 0)
    np.isin(${/data_01/surface_classification_flag}, [0, 1, 2])

"""

from __future__ import annotations

import ast
import logging
import posixpath
import re
from typing import TYPE_CHECKING

import numexpr
import numpy

if TYPE_CHECKING:
    import netCDF4

LOGGER = logging.getLogger(__name__)

# Regex to match ${/path/to/variable} references in expressions.
_VAR_REF_RE = re.compile(r"\$\{([^}]+)\}")


# ── Variable reference helpers ───────────────────────────────────────


def parse_variable_refs(expr: str) -> list[str]:
    """Extract variable paths from ``${...}`` references.

    Args:
        expr: Expression string with ``${/path}`` references.

    Returns:
        List of variable paths found in the expression (may contain
        duplicates; callers that need unique paths should deduplicate).

    """
    return _VAR_REF_RE.findall(expr)


def rewrite_expression(expr: str) -> tuple[str, dict[str, str]]:
    """Rewrite a user expression into a numexpr-ready string.

    Substitutes every ``${/path}`` reference with a placeholder
    identifier (``v0``, ``v1``, …) and translates the sole supported
    numpy helper ``np.isin(x, [a, b, …])`` into
    ``((x == a) | (x == b) | …)``.

    Args:
        expr: User-written expression with ``${...}`` syntax.

    Returns:
        Tuple ``(ne_expr, placeholders)`` where ``ne_expr`` is the
        numexpr-compatible expression and ``placeholders`` maps each
        placeholder name back to its variable path.

    """
    placeholders: dict[str, str] = {}
    path_to_name: dict[str, str] = {}

    def _substitute(match: re.Match[str]) -> str:
        path = match.group(1)
        if path not in path_to_name:
            name = f"v{len(path_to_name)}"
            path_to_name[path] = name
            placeholders[name] = path
        return path_to_name[path]

    substituted = _VAR_REF_RE.sub(_substitute, expr)
    # ``np.isin`` is the only non-numexpr helper actually used in
    # real expressions in this project; translate it before handing
    # the string to numexpr.
    rewritten = _rewrite_numpy_isin(substituted)
    return rewritten, placeholders


class _IsInTransformer(ast.NodeTransformer):
    """Rewrite ``np.isin(x, [a, b, c])`` → ``((x==a) | (x==b) | (x==c))``."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func = node.func
        is_isin = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "np"
            and func.attr == "isin"
        )
        if not is_isin or len(node.args) != 2:
            return node
        arr, values = node.args
        if not isinstance(values, ast.List | ast.Tuple) or not values.elts:
            return node

        def _eq(value: ast.expr) -> ast.Compare:
            return ast.Compare(left=arr, ops=[ast.Eq()], comparators=[value])

        result: ast.expr = _eq(values.elts[0])
        for elt in values.elts[1:]:
            result = ast.BinOp(left=result, op=ast.BitOr(), right=_eq(elt))
        return result


def _rewrite_numpy_isin(expr: str) -> str:
    """Replace ``np.isin(x, [a, b, c])`` calls with equivalent bitwise ORs.

    Operates on the AST to avoid regex pitfalls with nested brackets.
    If the string cannot be parsed the original expression is returned
    so numexpr can produce the appropriate error.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return expr
    transformed = _IsInTransformer().visit(tree)
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed)


def find_affected_groups(expr: str) -> set[str]:
    """Determine which top-level NetCDF groups are referenced.

    Args:
        expr: Expression with ``${/path}`` references.

    Returns:
        Set of top-level group names (e.g., ``{'data_01', 'data_20'}``).

    """
    refs = parse_variable_refs(expr)
    groups: set[str] = set()
    for ref in refs:
        # /data_01/altitude → parent is /data_01 → strip leading /
        parent = posixpath.dirname(ref).lstrip("/")
        if parent:
            # Keep only the top-level group (first component)
            groups.add(parent.split("/")[0])
    return groups


# ── Mask creation ────────────────────────────────────────────────────


def _load_variable(dataset: netCDF4.Dataset, path: str) -> numpy.ndarray:
    """Read a variable at a path like ``/data_01/altitude``."""
    return numpy.asarray(dataset[path][:])


def create_expression_mask(
    dataset: netCDF4.Dataset, expr: str
) -> tuple[numpy.ndarray | None, str | None]:
    """Create a boolean mask from a user expression.

    Accepts either a raw user expression containing ``${/path}``
    references or a pre-rewritten numexpr string.  When references
    are present the expression is rewritten internally so callers
    do not need to call :func:`rewrite_expression` first.

    Args:
        dataset: The NetCDF dataset to evaluate against.
        expr: The expression string.

    Returns:
        Tuple of ``(mask, None)`` on success or ``(None, error)``
        on failure.

    """
    if "${" in expr:
        ne_expr, placeholders = rewrite_expression(expr)
    else:
        ne_expr, placeholders = expr, {}

    try:
        local_dict = {
            name: _load_variable(dataset, path) for name, path in placeholders.items()
        }
    except Exception as exc:
        return None, f"Could not load variable for expression '{expr}': {exc}"

    try:
        result = numexpr.evaluate(ne_expr, local_dict=local_dict)
    except Exception as exc:
        return None, f"Error evaluating expression '{expr}': {exc}"

    if not isinstance(result, numpy.ndarray) or result.dtype != bool:
        return None, (
            f"Expression '{expr}' did not produce a boolean mask "
            f"(got dtype={getattr(result, 'dtype', type(result).__name__)})"
        )
    return result, None


# ── Variable listing ─────────────────────────────────────────────────


def list_all_variables(dataset: netCDF4.Dataset) -> list[str]:
    """List all variable paths in a NetCDF dataset.

    Args:
        dataset: The NetCDF dataset.

    Returns:
        Sorted list of variable paths (e.g., ``['/data_01/altitude', ...]``).

    """
    variables: list[str] = []

    def _walk_group(group: netCDF4.Dataset, prefix: str) -> None:
        variables.extend(f"{prefix}{varname}" for varname in sorted(group.variables))
        for grp_name, subgroup in sorted(group.groups.items()):
            _walk_group(subgroup, f"{prefix}{grp_name}/")

    _walk_group(dataset, "/")
    return variables


def list_all_groups(dataset: netCDF4.Dataset) -> list[str]:
    """List all group paths in a NetCDF dataset.

    Args:
        dataset: The NetCDF dataset.

    Returns:
        Sorted list of group paths (e.g., ``['/', '/data_01', ...]``).

    """
    groups: list[str] = ["/"]

    def _walk(group: netCDF4.Dataset, prefix: str) -> None:
        for grp_name, subgroup in sorted(group.groups.items()):
            path = f"{prefix}{grp_name}"
            groups.append(path)
            _walk(subgroup, f"{path}/")

    _walk(dataset, "/")
    return groups
