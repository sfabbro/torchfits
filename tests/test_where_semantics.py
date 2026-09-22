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
