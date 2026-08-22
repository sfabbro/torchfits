import pytest
import numpy as np
import pyarrow as pa
from torchfits._where import (
    _parse_where_literal,
    _tokenize_where_expression,
    _normalize_where_syntax,
    _parse_where_expression,
    parse_where_expression,
    _where_columns_from_ast,
)
from torchfits._table.read import _where_mask_for_table


def test_parse_where_literal():
    # Booleans
    assert _parse_where_literal("true") is True
    assert _parse_where_literal("TRUE") is True
    assert _parse_where_literal("false") is False
    assert _parse_where_literal("FALSE") is False

    # Nulls
    assert _parse_where_literal("none") is None
    assert _parse_where_literal("NONE") is None
    assert _parse_where_literal("null") is None
    assert _parse_where_literal("NULL") is None

    # Integers
    assert _parse_where_literal("42") == 42
    assert _parse_where_literal("-42") == -42
    assert _parse_where_literal("+42") == 42

    # Floats
    assert _parse_where_literal("3.14") == 3.14
    assert _parse_where_literal("-3.14") == -3.14
    assert _parse_where_literal(".5") == 0.5
    assert _parse_where_literal("1e3") == 1000.0
    assert _parse_where_literal("-1.5e-2") == -0.015

    # Quoted strings
    assert _parse_where_literal("'hello'") == "hello"
    assert _parse_where_literal('"world"') == "world"
    assert _parse_where_literal("'won\\'t'") == "won't"
    assert _parse_where_literal('"say \\"hi\\""') == 'say "hi"'

    # Bare words
    assert _parse_where_literal("STAR_A") == "STAR_A"
    assert _parse_where_literal("some_col") == "some_col"

    # Empty
    with pytest.raises(ValueError, match="where literal cannot be empty"):
        _parse_where_literal("")
    with pytest.raises(ValueError, match="where literal cannot be empty"):
        _parse_where_literal("   ")


def test_tokenize_where_expression():
    # Basic logic operators
    tokens = _tokenize_where_expression("A > 5 AND B <= 10")
    assert tokens == [
        ("WORD", "A"),
        ("OP", ">"),
        ("WORD", "5"),
        ("WORD", "AND"),
        ("WORD", "B"),
        ("OP", "<="),
        ("WORD", "10"),
    ]

    # Quotes and commas
    tokens = _tokenize_where_expression("NAME IN ('STAR_A', 'STAR_B')")
    assert tokens == [
        ("WORD", "NAME"),
        ("WORD", "IN"),
        ("LPAREN", "("),
        ("LITERAL", "STAR_A"),
        ("COMMA", ","),
        ("LITERAL", "STAR_B"),
        ("RPAREN", ")"),
    ]

    # Escape quotes
    tokens = _tokenize_where_expression("TEXT == 'It\\'s fine'")
    assert tokens == [
        ("WORD", "TEXT"),
        ("OP", "=="),
        ("LITERAL", "It's fine"),
    ]

    # Unterminated quotes
    with pytest.raises(
        ValueError, match="Unterminated quoted literal in where expression"
    ):
        _tokenize_where_expression("COL == 'missing quote")

    # Parentheses grouping
    tokens = _tokenize_where_expression("(X > 1) OR (Y < 2)")
    assert tokens == [
        ("LPAREN", "("),
        ("WORD", "X"),
        ("OP", ">"),
        ("WORD", "1"),
        ("RPAREN", ")"),
        ("WORD", "OR"),
        ("LPAREN", "("),
        ("WORD", "Y"),
        ("OP", "<"),
        ("WORD", "2"),
        ("RPAREN", ")"),
    ]


def test_normalize_where_syntax():
    # Test substitution of C-style operators
    assert _normalize_where_syntax("A && B") == "A  AND  B"
    assert _normalize_where_syntax("A || B") == "A  OR  B"

    # Bitwise/logical single char substitution with boundaries
    # Note: `(?<!\w)~(?!\w)` means ~ cannot be surrounded by word chars. So `~ A` or `~ (A)` works. `~A` does not.
    assert _normalize_where_syntax("~ A") == " NOT  A"
    assert _normalize_where_syntax("A & B") == "A  AND  B"
    assert _normalize_where_syntax("A | B") == "A  OR  B"

    # Compound cases
    assert (
        _normalize_where_syntax("~ A && (B || ~ (C)) | D & E")
        == " NOT  A  AND  (B  OR   NOT  (C))  OR  D  AND  E"
    )

    # Don't touch == or != or >= or <=
    assert _normalize_where_syntax("A == B && C != D") == "A == B  AND  C != D"


def test_normalize_where_syntax_quote_aware():
    # Operators inside quoted literals must survive verbatim.
    assert _normalize_where_syntax("NAME == 'AT&T'") == "NAME == 'AT&T'"
    assert _normalize_where_syntax("NAME == 'x||y'") == "NAME == 'x||y'"
    assert _normalize_where_syntax("A == 'A&&B' && B == 'C|D'") == (
        "A == 'A&&B'  AND  B == 'C|D'"
    )
    # FITS doubled-quote escapes become Python escapes.
    assert _normalize_where_syntax("NAME == 'O''NEIL'") == "NAME == 'O\\'NEIL'"
    # Unterminated quotes raise instead of silently mangling.
    with pytest.raises(ValueError, match="Unterminated"):
        _normalize_where_syntax("NAME == 'unterminated")


def test_parse_where_expression_quoted_literals():
    ast = parse_where_expression("NAME != 'AT&T' OR NAME IN ('A&B', 'C')")
    assert ast == (
        "or",
        ("cmp", "NAME", "!=", "AT&T"),
        ("in", "NAME", ["A&B", "C"], False),
    )
    assert parse_where_expression("NAME == 'O''NEIL'") == (
        "cmp",
        "NAME",
        "==",
        "O'NEIL",
    )


def test_parse_where_expression():
    # Simple comparisons
    assert _parse_where_expression("A > 5") == ("cmp", "A", ">", 5)
    assert _parse_where_expression("B == 'hello'") == ("cmp", "B", "==", "hello")

    # Logical AND / OR
    ast = _parse_where_expression("A > 5 AND B < 10")
    assert ast == ("and", ("cmp", "A", ">", 5), ("cmp", "B", "<", 10))

    ast = _parse_where_expression("A > 5 OR B < 10")
    assert ast == ("or", ("cmp", "A", ">", 5), ("cmp", "B", "<", 10))

    # NOT
    ast = _parse_where_expression("NOT A > 5")
    assert ast == ("not", ("cmp", "A", ">", 5))

    # Precedence AND over OR
    ast = _parse_where_expression("A > 1 OR B > 2 AND C > 3")
    assert ast == (
        "or",
        ("cmp", "A", ">", 1),
        ("and", ("cmp", "B", ">", 2), ("cmp", "C", ">", 3)),
    )

    # Parentheses override precedence
    ast = _parse_where_expression("(A > 1 OR B > 2) AND C > 3")
    assert ast == (
        "and",
        ("or", ("cmp", "A", ">", 1), ("cmp", "B", ">", 2)),
        ("cmp", "C", ">", 3),
    )

    # IN / NOT IN
    assert _parse_where_expression("X IN (1, 2, 3)") == ("in", "X", [1, 2, 3], False)
    assert _parse_where_expression("Y NOT IN ('a', 'b')") == (
        "in",
        "Y",
        ["a", "b"],
        True,
    )

    # BETWEEN / NOT BETWEEN
    assert _parse_where_expression("Z BETWEEN 1 AND 10") == (
        "between",
        "Z",
        1,
        10,
        False,
    )
    assert _parse_where_expression("W NOT BETWEEN 0.5 AND 1.5") == (
        "between",
        "W",
        0.5,
        1.5,
        True,
    )

    # IS NULL / IS NOT NULL
    assert _parse_where_expression("COL IS NULL") == ("isnull", "COL", False)
    assert _parse_where_expression("COL IS NOT NULL") == ("isnull", "COL", True)
    assert _parse_where_expression("COL NOT NULL") == ("isnull", "COL", True)

    # Errors
    with pytest.raises(ValueError, match="where must be a non-empty string"):
        _parse_where_expression("")
    with pytest.raises(ValueError, match="where expects a column identifier"):
        _parse_where_expression("5 > A")  # Left side must be a column name
    with pytest.raises(
        ValueError, match="(Unbalanced parentheses|Unexpected end of where expression)"
    ):
        _parse_where_expression("(A > 5")
    with pytest.raises(
        ValueError, match="(Unbalanced parentheses|Unexpected trailing tokens)"
    ):
        _parse_where_expression("(A > 5))")
    with pytest.raises(ValueError, match="Unexpected trailing tokens"):
        _parse_where_expression("A > 5 B < 10")


def test_where_columns_from_ast():
    # Simple
    ast = _parse_where_expression("A > 5")
    assert _where_columns_from_ast(ast) == ["A"]

    # Multiple columns, unique extraction
    ast = _parse_where_expression("A > 5 AND B < 10 OR A == 2")
    cols = _where_columns_from_ast(ast)
    assert set(cols) == {"A", "B"}
    assert len(cols) == 2

    # Various types of nodes
    ast = _parse_where_expression("X IN (1, 2) OR Y BETWEEN 1 AND 2 OR Z IS NULL")
    cols = _where_columns_from_ast(ast)
    assert set(cols) == {"X", "Y", "Z"}


def _where_mask(data: dict, condition: str) -> np.ndarray:
    """Build Arrow table from data dict and run WHERE via pyarrow.compute."""
    pa_arrays = {}
    for name, arr in data.items():
        if isinstance(arr, pa.Array) or isinstance(arr, pa.ChunkedArray):
            pa_arrays[name] = arr
        elif isinstance(arr, np.ndarray) and arr.dtype.kind == "O":
            pa_arrays[name] = pa.array(arr.tolist())
        else:
            pa_arrays[name] = pa.array(arr)
    pa_table = pa.table(pa_arrays)
    mask = _where_mask_for_table(pa_table, condition)
    return mask.to_numpy()


def test__evaluate_where():
    data = {
        "A": np.array([1, 2, 3, 4, 5]),
        "B": np.array([10, 20, 30, 40, 50]),
        "C": np.array(["foo", "bar", "baz", "qux", "quux"]),
        "E": np.array([None, "test", None, "data", "point"], dtype=object),
    }

    # Comparison
    mask = _where_mask(data, "A > 2")
    np.testing.assert_array_equal(mask, [False, False, True, True, True])

    mask = _where_mask(data, "A <= 3")
    np.testing.assert_array_equal(mask, [True, True, True, False, False])

    mask = _where_mask(data, "C == 'baz'")
    np.testing.assert_array_equal(mask, [False, False, True, False, False])

    mask = _where_mask(data, "C != 'baz'")
    np.testing.assert_array_equal(mask, [True, True, False, True, True])

    # AND / OR
    mask = _where_mask(data, "A > 1 AND B < 50")
    np.testing.assert_array_equal(mask, [False, True, True, True, False])

    mask = _where_mask(data, "A == 1 OR A == 5")
    np.testing.assert_array_equal(mask, [True, False, False, False, True])

    # NOT
    mask = _where_mask(data, "NOT A > 2")
    np.testing.assert_array_equal(mask, [True, True, False, False, False])

    # IN
    mask = _where_mask(data, "A IN (2, 4)")
    np.testing.assert_array_equal(mask, [False, True, False, True, False])

    mask = _where_mask(data, "A NOT IN (2, 4)")
    np.testing.assert_array_equal(mask, [True, False, True, False, True])

    # BETWEEN
    mask = _where_mask(data, "A BETWEEN 2 AND 4")
    np.testing.assert_array_equal(mask, [False, True, True, True, False])

    mask = _where_mask(data, "A NOT BETWEEN 2 AND 4")
    np.testing.assert_array_equal(mask, [True, False, False, False, True])

    # IS NULL — float nulls (Arrow uses None, not NaN, for null)
    float_nulls = pa.array([1.1, None, 3.3, 4.4, None], type=pa.float64())
    mask = _where_mask({"F": float_nulls}, "F IS NULL")
    np.testing.assert_array_equal(mask, [False, True, False, False, True])

    mask = _where_mask({"F": float_nulls}, "F IS NOT NULL")
    np.testing.assert_array_equal(mask, [True, False, True, True, False])

    # IS NULL — E is object with None
    mask = _where_mask(data, "E IS NULL")
    np.testing.assert_array_equal(mask, [True, False, True, False, False])

    # Missing column
    with pytest.raises(ValueError, match="unknown column"):
        _where_mask(data, "MISSING > 0")

    # == NULL and != NULL
    mask = _where_mask(data, "E == NULL")
    np.testing.assert_array_equal(mask, [True, False, True, False, False])

    mask = _where_mask(data, "E != NULL")
    np.testing.assert_array_equal(
        mask, [False, True, False, True, True]
    )  # NULL comparisons: Arrow rejects > NULL gracefully
    with pytest.raises(
        ValueError, match="where comparisons with null only support == and !="
    ):
        _where_mask(data, "E > NULL")
