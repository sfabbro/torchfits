"""Tests for table read backend / where strategy policy."""

from __future__ import annotations

from torchfits._table_engine.read_policy import (
    WhereStrategy,
    choose_where_read_plan,
    should_skip_cpp_numpy_for_where,
)


def test_should_skip_cpp_numpy_for_where():
    assert should_skip_cpp_numpy_for_where("auto", "MAG < 20") is True
    assert should_skip_cpp_numpy_for_where("auto", None) is False
    assert should_skip_cpp_numpy_for_where("cpp_numpy", "MAG < 20") is False


def test_choose_where_read_plan_small_table():
    header = {"TFIELDS": 1, "TTYPE1": "MAG", "TFORM1": "1E", "NAXIS2": 50}
    plan = choose_where_read_plan(
        header=header,
        header_ok=True,
        columns=["MAG"],
        backend="auto",
        n_rows=50,
        env={},
    )
    assert plan.strategy == WhereStrategy.ARROW_FILTER
    assert plan.unfiltered_backend == "cpp_numpy"


def test_choose_where_read_plan_large_table():
    header = {"TFIELDS": 1, "TTYPE1": "MAG", "TFORM1": "1E", "NAXIS2": 500_000}
    plan = choose_where_read_plan(
        header=header,
        header_ok=True,
        columns=["MAG"],
        backend="auto",
        n_rows=500_000,
        env={},
    )
    assert plan.strategy == WhereStrategy.CPP_PUSHDOWN
