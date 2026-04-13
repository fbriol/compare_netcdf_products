"""Safe expression evaluation for filtering NetCDF data.

Expression syntax uses ``${/path/to/variable}`` to reference NetCDF
variables.  Before evaluation the references are rewritten to standard
``ds['/path/to/variable'][:]`` calls so that the :class:`Expression`
evaluator can process them.

Examples::

    ${/data_20/distance_to_coast} > 30000
    (${/data_20/distance_to_coast} > 30000) & (${/data_20/flag} == 0)
    np.isin(${/data_01/surface_classification_flag}, [0, 1, 2])

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
import ast
import dataclasses
import logging
import posixpath
import re

import numpy

if TYPE_CHECKING:
    import netCDF4

LOGGER = logging.getLogger(__name__)

# Regex to match ${/path/to/variable} references in expressions.
_VAR_REF_RE = re.compile(r'\$\{([^}]+)\}')


class NumpyWrapper:
    """Wrapper that allows secure access to all numpy attributes."""

    def __getattr__(self, name: str) -> object:
        """Dynamically access numpy attributes."""
        if hasattr(numpy, name):
            attr = getattr(numpy, name)
            if callable(attr) or isinstance(
                    attr, (int, float, complex, bool, numpy.number)):
                return attr
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")


@dataclasses.dataclass
class Expression:
    """Partitioning expressions.

    Args:
        expression: The expression to be evaluated.

    Raises:
        NameError: If a variable is not defined.

    """

    __slots__ = ("code", "expr")

    code: Any
    expr: str

    BUILTINS: ClassVar[dict[str, Any]] = {
        "range": range,
        "__import__": __import__,
        "len": len,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "any": any,
        "all": all,
        "np": NumpyWrapper(),
    }

    def __init__(self, expr: str) -> None:
        self.expr = expr
        self.code = compile(ast.parse(expr, mode="eval"), " ", "eval")

    def __call__(self, variables: dict[str, Any]) -> object:
        try:
            parsed = ast.parse(self.expr, mode="eval")
            variable_names = set()
            builtins_set = self.BUILTINS

            class NameExtractor(ast.NodeVisitor):

                def visit_Name(self, node: ast.Name) -> None:
                    if node.id not in builtins_set:
                        variable_names.add(node.id)
                    self.generic_visit(node)

            NameExtractor().visit(parsed)

            __locals = {name: variables[name] for name in variable_names}
            return eval(self.code, {"__builtins__": self.BUILTINS}, __locals)
        except KeyError as err:
            raise NameError(f"name {err!s} is not defined") from err


# ── Variable reference helpers ───────────────────────────────────────


def parse_variable_refs(expr: str) -> list[str]:
    """Extract variable paths from ``${...}`` references.

    Args:
        expr: Expression string with ``${/path}`` references.

    Returns:
        List of variable paths found in the expression.

    """
    return _VAR_REF_RE.findall(expr)


def rewrite_expression(expr: str) -> str:
    """Rewrite ``${/path}`` references to ``ds['/path'][:]``.

    Args:
        expr: User-written expression with ``${...}`` syntax.

    Returns:
        Expression ready for the :class:`Expression` evaluator.

    """
    return _VAR_REF_RE.sub(r"ds['\1'][:]", expr)


def find_affected_groups(expr: str) -> set[str]:
    """Determine which NetCDF groups are affected by the expression.

    Extracts the group paths from the variable references in the
    expression.  No hardcoded group list is needed.

    Args:
        expr: Expression with ``${/path}`` references.

    Returns:
        Set of group paths (e.g., ``{'data_01', 'data_20'}``).

    """
    refs = parse_variable_refs(expr)
    groups: set[str] = set()
    for ref in refs:
        # /data_01/altitude → parent is /data_01 → strip leading /
        parent = posixpath.dirname(ref).lstrip("/")
        if parent:
            # Keep only the top-level group (first component)
            top_level = parent.split("/")[0]
            groups.add(top_level)
    return groups


# ── Mask creation ────────────────────────────────────────────────────


def create_expression_mask(
        dataset: netCDF4.Dataset,
        expr: str) -> tuple[numpy.ndarray | None, str | None]:
    """Create a boolean mask from an expression.

    The *expr* should already be in evaluable form (i.e. the
    ``${...}`` references must have been rewritten via
    :func:`rewrite_expression` before calling this function).

    Args:
        dataset: The NetCDF dataset to evaluate against.
        expr: The rewritten expression string.

    Returns:
        Tuple of ``(mask, None)`` on success or ``(None, error)``
        on failure.

    """
    try:
        expression = Expression(expr)
    except SyntaxError as e:
        return None, f"Invalid expression '{expr}': {e}"

    var_data = {'ds': dataset}
    try:
        mask = expression(var_data)
        return mask, None
    except Exception as e:
        return None, f"Error evaluating expression '{expr}': {e}"


# ── Variable listing ─────────────────────────────────────────────────


def list_all_variables(dataset: netCDF4.Dataset) -> list[str]:
    """List all variable paths in a NetCDF dataset.

    Args:
        dataset: The NetCDF dataset.

    Returns:
        Sorted list of variable paths (e.g., ``['/data_01/altitude', ...]``).

    """
    variables: list[str] = []

    def _walk_group(group: netCDF4.Group, prefix: str) -> None:
        for varname in sorted(group.variables):
            variables.append(f"{prefix}{varname}")
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

    def _walk(group: netCDF4.Group, prefix: str) -> None:
        for grp_name, subgroup in sorted(group.groups.items()):
            path = f"{prefix}{grp_name}"
            groups.append(path)
            _walk(subgroup, f"{path}/")

    _walk(dataset, "/")
    return groups
