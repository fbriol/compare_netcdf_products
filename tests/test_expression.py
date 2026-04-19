"""Tests for the expression module."""

from __future__ import annotations

import numpy

from app.expression import (
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
    def test_simple_ref_becomes_placeholder(self):
        ne_expr, placeholders = rewrite_expression("${/data_01/alt} > 100")
        assert ne_expr == "v0 > 100"
        assert placeholders == {"v0": "/data_01/alt"}

    def test_multiple_refs_get_distinct_placeholders(self):
        ne_expr, placeholders = rewrite_expression("${/a/b} + ${/c/d}")
        assert ne_expr == "v0 + v1"
        assert placeholders == {"v0": "/a/b", "v1": "/c/d"}

    def test_duplicate_ref_reuses_placeholder(self):
        ne_expr, placeholders = rewrite_expression("${/x} + ${/x}")
        assert ne_expr == "v0 + v0"
        assert placeholders == {"v0": "/x"}

    def test_no_refs_unchanged(self):
        ne_expr, placeholders = rewrite_expression("x + 1")
        assert ne_expr == "x + 1"
        assert placeholders == {}

    def test_np_isin_is_expanded(self):
        ne_expr, placeholders = rewrite_expression("np.isin(${/data/flag}, [0, 1, 2])")
        assert placeholders == {"v0": "/data/flag"}
        # Order of OR arms and exact parens may vary; check semantics.
        # Verify it eval's correctly via numexpr.
        import numexpr

        arr = numpy.array([0, 1, 2, 3, 4])
        result = numexpr.evaluate(ne_expr, local_dict={"v0": arr})
        numpy.testing.assert_array_equal(
            result, numpy.array([True, True, True, False, False])
        )


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


# ── create_expression_mask ────────────────────────────────────────────


class TestCreateExpressionMask:
    def test_valid_expression(self, open_nc_pair):
        src_ds, _ = open_nc_pair
        mask, error = create_expression_mask(src_ds, "${/data_01/flag} == 0")
        assert error is None
        assert mask is not None
        assert isinstance(mask, numpy.ndarray)
        assert mask.dtype == bool
        assert mask.all()  # all flags are 0 in fixture

    def test_combined_expression(self, open_nc_pair):
        src_ds, _ = open_nc_pair
        mask, error = create_expression_mask(
            src_ds,
            "(${/data_01/altitude} > -1000) & (${/data_01/flag} == 0)",
        )
        assert error is None
        assert mask is not None
        assert mask.dtype == bool

    def test_np_isin_expression(self, open_nc_pair):
        src_ds, _ = open_nc_pair
        mask, error = create_expression_mask(
            src_ds, "np.isin(${/data_01/flag}, [0, 1, 2])"
        )
        assert error is None
        assert mask is not None
        assert mask.dtype == bool
        assert mask.all()

    def test_invalid_syntax(self, open_nc_pair):
        src_ds, _ = open_nc_pair
        mask, error = create_expression_mask(src_ds, "invalid syntax !!!")
        assert mask is None
        assert error is not None

    def test_unknown_variable_ref(self, open_nc_pair):
        src_ds, _ = open_nc_pair
        mask, error = create_expression_mask(src_ds, "${/does/not/exist} > 0")
        assert mask is None
        assert error is not None

    def test_non_boolean_result_rejected(self, open_nc_pair):
        src_ds, _ = open_nc_pair
        mask, error = create_expression_mask(src_ds, "${/data_01/altitude} + 1")
        assert mask is None
        assert error is not None and "boolean" in error.lower()

    def test_import_is_blocked(self, open_nc_pair):
        # numexpr's parser rejects attribute access and Python builtins,
        # so the historical RCE vector cannot even parse.
        src_ds, _ = open_nc_pair
        mask, error = create_expression_mask(
            src_ds, "__import__('os').system('echo pwned')"
        )
        assert mask is None
        assert error is not None

    def test_attribute_access_is_blocked(self, open_nc_pair):
        src_ds, _ = open_nc_pair
        mask, error = create_expression_mask(src_ds, "(1).__class__")
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
