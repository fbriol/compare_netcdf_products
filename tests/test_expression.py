"""Tests for the expression module."""

from __future__ import annotations

import numpy
import pytest

from app.expression import (
    Expression,
    create_expression_mask,
    find_affected_groups,
    list_all_groups,
    list_all_variables,
    parse_variable_refs,
    rewrite_expression,
)

# ── parse_variable_refs ───────────────────────────────────────────────


class TestParseVariableRefs:
    def test_single_ref(self):
        refs = parse_variable_refs("${/data_01/altitude} > 100")
        assert refs == ["/data_01/altitude"]

    def test_multiple_refs(self):
        expr = "(${/data_01/alt} > 0) & (${/data_20/flag} == 1)"
        refs = parse_variable_refs(expr)
        assert set(refs) == {"/data_01/alt", "/data_20/flag"}

    def test_no_refs(self):
        refs = parse_variable_refs("x > 10")
        assert refs == []

    def test_nested_path(self):
        refs = parse_variable_refs("${/group/sub/var} < 5")
        assert refs == ["/group/sub/var"]


# ── rewrite_expression ────────────────────────────────────────────────


class TestRewriteExpression:
    def test_simple(self):
        result = rewrite_expression("${/data_01/alt} > 100")
        assert result == "ds['/data_01/alt'][:] > 100"

    def test_multiple(self):
        expr = "${/a/b} + ${/c/d}"
        result = rewrite_expression(expr)
        assert result == "ds['/a/b'][:] + ds['/c/d'][:]"

    def test_no_refs_unchanged(self):
        result = rewrite_expression("x + 1")
        assert result == "x + 1"


# ── find_affected_groups ──────────────────────────────────────────────


class TestFindAffectedGroups:
    def test_single_group(self):
        groups = find_affected_groups("${/data_01/altitude} > 100")
        assert groups == {"data_01"}

    def test_multiple_groups(self):
        expr = "${/data_01/alt} > 0 & ${/data_20/flag} == 0"
        groups = find_affected_groups(expr)
        assert groups == {"data_01", "data_20"}

    def test_nested_group_uses_top_level(self):
        groups = find_affected_groups("${/group/sub/var} < 5")
        assert groups == {"group"}

    def test_no_refs_empty(self):
        groups = find_affected_groups("x > 10")
        assert groups == set()


# ── Expression evaluator ──────────────────────────────────────────────


class TestExpression:
    def test_simple_eval(self):
        expr = Expression("x > 2")
        result = expr({"x": numpy.array([1, 2, 3, 4])})
        expected = numpy.array([False, False, True, True])
        numpy.testing.assert_array_equal(result, expected)

    def test_numpy_functions(self):
        expr = Expression("np.isin(x, [1, 3])")
        result = expr({"x": numpy.array([1, 2, 3, 4])})
        expected = numpy.array([True, False, True, False])
        numpy.testing.assert_array_equal(result, expected)

    def test_undefined_variable_raises(self):
        expr = Expression("x + y")
        with pytest.raises(NameError):
            expr({"x": numpy.array([1])})

    def test_syntax_error(self):
        with pytest.raises(SyntaxError):
            Expression("if True")

    def test_builtins_available(self):
        expr = Expression("len(x)")
        result = expr({"x": [1, 2, 3]})
        assert result == 3

    def test_import_is_blocked(self):
        with pytest.raises((SyntaxError, NameError)):
            Expression("__import__('os').system('echo pwned')")

    def test_dunder_attribute_is_blocked(self):
        with pytest.raises(SyntaxError):
            Expression("np.__class__")

    def test_numpy_private_attribute_is_blocked(self):
        expr = Expression("np.core")
        with pytest.raises(AttributeError):
            expr({})

    def test_lambda_is_blocked(self):
        with pytest.raises(SyntaxError):
            Expression("(lambda: 1)()")


# ── create_expression_mask ────────────────────────────────────────────


class TestCreateExpressionMask:
    def test_valid_expression(self, open_nc_pair):
        src_ds, _ = open_nc_pair
        rewritten = rewrite_expression("${/data_01/flag} == 0")
        mask, error = create_expression_mask(src_ds, rewritten)
        assert error is None
        assert mask is not None
        assert isinstance(mask, numpy.ndarray)
        assert mask.all()  # all flags are 0

    def test_invalid_expression(self, open_nc_pair):
        src_ds, _ = open_nc_pair
        mask, error = create_expression_mask(src_ds, "invalid syntax !!!")
        assert mask is None
        assert error is not None


# ── list_all_variables ────────────────────────────────────────────────


class TestListAllVariables:
    def test_lists_variables(self, open_nc_pair):
        src_ds, _ = open_nc_pair
        variables = list_all_variables(src_ds)
        assert "/data_01/altitude" in variables
        assert "/data_01/speed" in variables
        assert len(variables) >= 2


# ── list_all_groups ───────────────────────────────────────────────────


class TestListAllGroups:
    def test_lists_groups(self, open_nc_pair):
        src_ds, _ = open_nc_pair
        groups = list_all_groups(src_ds)
        assert "/" in groups
        assert "/data_01" in groups
