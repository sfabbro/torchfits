"""Row-set semantics of the public ``torchfits.where`` predicate helpers.

Three-valued logic contract (SQL-style, NaN for floats / ``None`` for object
arrays as the null convention): negations and ``!=`` must not resurrect null
rows, ``NOT (X == v)`` must equal ``X != v``, and keyword rewrites must never
touch quoted string literals.
"""

import numpy as np
import pytest

from torchfits import where


def test_neq_and_not_agree_excluding_nan():
    """NOT (X == 5) and X != 5 must select identical rows on NaN data."""
    x = np.array([1.0, 5.0, float("nan")])
    neq = where.evaluate_where(where.parse_where_expression("X != 5"), {"X": x})
    neg = where.evaluate_where(where.parse_where_expression("NOT X == 5"), {"X": x})
    np.testing.assert_array_equal(neq, [True, False, False])
    np.testing.assert_array_equal(neq, neg)


def test_negated_in_and_between_exclude_nan():
    x = np.array([1.0, 5.0, float("nan")])
    not_in = where.evaluate_where(
        where.parse_where_expression("X NOT IN (5)"), {"X": x}
    )
    not_between = where.evaluate_where(
        where.parse_where_expression("X NOT BETWEEN 0 AND 2"), {"X": x}
    )
    np.testing.assert_array_equal(not_in, [True, False, False])
    np.testing.assert_array_equal(not_between, [False, True, False])


def test_object_none_is_null_like_for_negation():
    """None rows are null rows: IS NULL sees them, negations drop them."""
    o = np.array([1, None, 3], dtype=object)
    isnull = where.evaluate_where(where.parse_where_expression("O IS NULL"), {"O": o})
    np.testing.assert_array_equal(isnull, [False, True, False])
    neq = where.evaluate_where(where.parse_where_expression("O != 1"), {"O": o})
    neg = where.evaluate_where(where.parse_where_expression("NOT O == 1"), {"O": o})
    np.testing.assert_array_equal(neq, [False, False, True])
    np.testing.assert_array_equal(neq, neg)


def test_quoted_null_words_stay_strings():
    """'none' / 'null' quoted are string literals; only bare words are NULL."""
    assert where.parse_where_expression("NAME == 'none'") == (
        "cmp",
        "NAME",
        "==",
        "none",
    )
    assert where.parse_where_expression("NAME == 'null'") == (
        "cmp",
        "NAME",
        "==",
        "null",
    )
    assert where.parse_where_expression("NAME IN ('NULL', 'x')") == (
        "in",
        "NAME",
        ["NULL", "x"],
        False,
    )
    # Bare words keep the NULL convention.
    assert where.parse_where_expression("NAME == none") == ("cmp", "NAME", "==", None)


def test_keyword_rewrites_do_not_touch_quoted_literals():
    """BETWEEN / IS NULL / AND rewrites are syntax, not string content."""
    assert where.parse_where_expression("NAME == 'a BETWEEN 1 AND 2'") == (
        "cmp",
        "NAME",
        "==",
        "a BETWEEN 1 AND 2",
    )
    assert where.parse_where_expression("NAME == 'x IS NULL'") == (
        "cmp",
        "NAME",
        "==",
        "x IS NULL",
    )
    assert where.parse_where_expression("NAME == 'O''NEIL AND x'") == (
        "cmp",
        "NAME",
        "==",
        "O'NEIL AND x",
    )
    mask = where.evaluate_where(
        where.parse_where_expression("NAME == 'a BETWEEN 1 AND 2'"),
        {"NAME": np.array(["a BETWEEN 1 AND 2", "a"], dtype=object)},
    )
    np.testing.assert_array_equal(mask, [True, False])


def test_evaluate_where_wraps_bad_comparisons_as_value_error():
    """Incomparable literal/column pairs raise ValueError, never numpy errors."""
    data = {"A": np.array([1, 2, 3])}
    for expr in ("A == 'zzz'", "A != 'zzz'", "A > 'zzz'", "A BETWEEN 'a' AND 'b'"):
        with pytest.raises(ValueError):
            where.evaluate_where(where.parse_where_expression(expr), data)


# --------------------------------------------------------------------------
# FITS / Arrow table read path -- backend and evaluator parity
#
# These three tests do not name a dialect: they assert that whichever rule the
# two engines follow, they follow the *same* one. They are what found the
# original disagreement between the NumPy reference and the Arrow engine, and
# they hold under either null contract.
# --------------------------------------------------------------------------


def _tnull_table_file():
    """3-row table whose column A is NULL in the middle row (TNULL sentinel)."""
    import tempfile

    from astropy.io import fits
    from astropy.table import Table

    table = Table(
        {
            "ID": np.array([0, 1, 2], dtype=np.int32),
            "A": np.array([1, -999, 3], dtype=np.int16),
        }
    )
    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()
    table.write(handle.name, format="fits", overwrite=True)
    with fits.open(handle.name, mode="update") as hdul:
        # Column order is ID=1, A=2, so the sentinel belongs to TNULL2.
        hdul[1].header["TNULL2"] = -999
    return handle.name


def _selected_rows(path, where_expr, **kwargs):
    import torchfits

    return sorted(
        torchfits.table.read(path, hdu=1, where=where_expr, **kwargs)["ID"].to_pylist()
    )


@pytest.fixture
def tnull_path():
    path = _tnull_table_file()
    try:
        # Guard the fixture: every assertion below is meaningless without a
        # genuine null row, and a mistargeted TNULL card is silent.
        import torchfits

        full = torchfits.table.read(path, hdu=1)
        assert full["A"].is_null().to_pylist() == [False, True, False]
        yield path
    finally:
        import os

        os.unlink(path)


@pytest.mark.parametrize("backend", ["auto", "cpp", "torch"])
@pytest.mark.parametrize(
    "expr",
    [
        "A == 1",
        "A != 1",
        "A > 1",
        "A <= 1",
        "A IN (1)",
        "A NOT IN (1)",
        "A BETWEEN 0 AND 2",
        "NOT (A == 1)",
        "A IS NULL",
    ],
)
def test_every_backend_agrees_on_the_null_contract(tnull_path, backend, expr):
    """The Arrow engine, the C++ pushdown and the torch mask must not diverge.

    Each of the three evaluates nulls somewhere different -- Arrow resolves
    them in the mask, the C++ pushdown compares raw stored sentinels, and the
    torch path sees NaN -- so a null row is where they are most likely to
    part company.
    """
    assert _selected_rows(tnull_path, expr, backend=backend) == _selected_rows(
        tnull_path, expr
    )


def test_where_evaluator_and_arrow_path_agree_on_null_rows(tnull_path):
    """torchfits.where (NumPy) and table.read (Arrow) share one contract.

    The reference evaluator reads an object array with ``None``; the table
    path reads the same logical column as a TNULL sentinel. The selected rows
    must match expression for expression.
    """
    data = np.array([1, None, 3], dtype=object)
    for expr in (
        "A != 1",
        "NOT (A == 1)",
        "A == 1",
        "A IS NULL",
        "A IS NOT NULL",
        "A IN (1)",
        "A NOT IN (1)",
    ):
        ref = where.evaluate_where(where.parse_where_expression(expr), {"A": data})
        ref_rows = sorted(i for i, v in enumerate(np.asarray(ref, dtype=object)) if v)
        assert _selected_rows(tnull_path, expr) == ref_rows, expr


BACKENDS = ("auto", "cpp", "torch")


@pytest.mark.parametrize(
    "negated, equivalent",
    [
        ("NOT (A == 1)", "A != 1"),
        ("NOT (A == 5)", "A != 5"),
        ("NOT (A IN (1))", "A NOT IN (1)"),
        ("NOT (A IS NULL)", "A IS NOT NULL"),
        ("NOT NOT (A == 1)", "A == 1"),
        ("NOT (A == 1 AND A == 3)", "A != 1 OR A != 3"),
        ("NOT (A == 1 OR A == 3)", "A != 1 AND A != 3"),
    ],
)
def test_negation_equals_its_dual_in_every_engine(tnull_path, negated, equivalent):
    """One rule, four evaluators: the NumPy reference, Arrow, C++ and torch.

    Each of the four resolves nulls somewhere different -- the reference masks
    them out of the negated result, Arrow keeps a null mask until the top of
    the tree, the C++ pushdown compares raw sentinels, and the torch path sees
    NaN -- so a negated predicate over a null column is where they part company.
    """
    data = np.array([1, None, 3], dtype=object)

    def reference(expr):
        mask = where.evaluate_where(where.parse_where_expression(expr), {"A": data})
        return sorted(i for i, v in enumerate(np.asarray(mask, dtype=object)) if v)

    for expr in (negated, equivalent):
        assert _selected_rows(tnull_path, expr, backend="auto") == reference(expr), expr
        for backend in BACKENDS:
            assert _selected_rows(tnull_path, expr, backend=backend) == reference(
                expr
            ), (expr, backend)
    assert _selected_rows(tnull_path, negated) == _selected_rows(tnull_path, equivalent)


@pytest.mark.parametrize(
    "negated, equivalent",
    [
        ("NOT (A BETWEEN 0 AND 2)", "A NOT BETWEEN 0 AND 2"),
        ("NOT (A > 1)", "A <= 1"),
        ("NOT (A >= 1)", "A < 1"),
        ("NOT (A < 3)", "A >= 3"),
        ("NOT (A <= 1)", "A > 1"),
    ],
)
def test_negation_equals_its_dual_for_ordering_and_between(
    tnull_path, negated, equivalent
):
    """The dual-operator identity also holds for the ordering and BETWEEN forms.

    These are excluded from the test above because the NumPy reference cannot
    evaluate them at all: numpy object ordering raises `TypeError` on `None`,
    which ``evaluate_where`` wraps in ``ValueError``. The three read backends
    can, so they are compared against each other.
    """
    expected = _selected_rows(tnull_path, negated)
    assert expected == _selected_rows(tnull_path, equivalent)
    for backend in BACKENDS:
        assert _selected_rows(tnull_path, negated, backend=backend) == expected


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("A == 1", [0]),
        ("A > 1", [2]),
        ("A <= 1", [0]),
        ("A IN (1)", [0]),
        ("A BETWEEN 0 AND 2", [0]),
        ("A != 1", [2]),
        ("A NOT IN (1)", [2]),
        ("A NOT BETWEEN 0 AND 2", [2]),
        ("A IS NULL", [1]),
        ("A IS NOT NULL", [0, 2]),
    ],
)
def test_null_rows_follow_sql_three_valued_logic(tnull_path, expr, expected):
    """A null is unknown: no positive test selects it, no negative one rescues it.

    Row 1 is the null row, so every positive predicate must omit it and every
    negated one must omit it too. ``IS NULL`` / ``IS NOT NULL`` are the way to
    test for absence.
    """
    for backend in BACKENDS:
        assert _selected_rows(tnull_path, expr, backend=backend) == expected, (
            expr,
            backend,
        )
