"""Public table-predicate parsing and evaluation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, cast

if TYPE_CHECKING:
    import numpy as np

from ._where import (
    _WHERE_IDENT_RE as where_identifier_re,
    _normalize_where_syntax as normalize_where_syntax,
    _parse_where_expression as parse_where_expression,
    _parse_where_literal as parse_where_literal,
    _tokenize_where_expression as tokenize_where_expression,
    _where_columns_from_ast as where_columns_from_ast,
)


def _not_null_like(values: "np.ndarray") -> "np.ndarray":
    """Boolean mask of positions that are not null-like (NaN for floats)."""
    import numpy as np

    if np.issubdtype(values.dtype, np.floating):
        return ~np.isnan(values)
    return np.ones(values.shape, dtype=bool)


def evaluate_where(ast: tuple[Any, ...], data: Mapping[str, Any]) -> np.ndarray:
    """Evaluate a parsed predicate against mapping values as NumPy arrays.

    The ``numpy`` import is lazy to avoid a mandatory dependency at
    the package level (torchfits itself only requires PyTorch).

    Null checks: use ``isnull`` / ``notnull`` in the where grammar (or
    ``table.read(..., where=)``, which uses Arrow ``pc.is_null``). Comparing
    with ``== NULL`` / ``!= NULL`` on numeric arrays is rejected — FITS nulls
    are TNULLn / NaN, not Python ``None``.
    """
    import numpy as np

    kind = ast[0]
    if kind == "and":
        return cast(
            np.ndarray, evaluate_where(ast[1], data) & evaluate_where(ast[2], data)
        )
    if kind == "or":
        return cast(
            np.ndarray, evaluate_where(ast[1], data) | evaluate_where(ast[2], data)
        )
    if kind == "not":
        inverted = ~evaluate_where(ast[1], data)
        # Exclude null-like positions from negated results so
        # NOT (X == 5) stays equivalent to X != 5 (NaN rows excluded).
        keep = None
        for name in where_columns_from_ast(ast[1]):
            if name in data:
                finite = _not_null_like(np.asarray(data[name]))
                keep = finite if keep is None else (keep & finite)
        return cast(np.ndarray, inverted if keep is None else (inverted & keep))

    column = ast[1]
    if column not in data:
        raise ValueError(f"Unknown column: {column}")
    values = np.asarray(data[column])
    if kind == "cmp":
        _, _, operator, literal = ast
        if literal is None:
            if np.issubdtype(values.dtype, np.number) or np.issubdtype(
                values.dtype, np.bool_
            ):
                raise ValueError(
                    "COL == NULL / != NULL is not supported on numeric arrays "
                    "(FITS uses TNULLn / NaN). Use isnull/notnull, or "
                    "table.read(..., where=) for FITS-native null handling."
                )
            if operator == "==":
                return np.asarray([value is None for value in values], dtype=bool)
            if operator == "!=":
                return np.asarray([value is not None for value in values], dtype=bool)
            raise ValueError("NULL comparisons only support == and !=")
        operators = {
            "==": np.equal,
            "!=": np.not_equal,
            ">": np.greater,
            ">=": np.greater_equal,
            "<": np.less,
            "<=": np.less_equal,
        }
        try:
            return cast(np.ndarray, operators[operator](values, literal))
        except KeyError as exc:
            raise ValueError(f"Unsupported operator: {operator}") from exc
    if kind == "in":
        _, _, literals, negate = ast
        mask = cast(np.ndarray, np.isin(values, literals))
        if negate:
            return cast(np.ndarray, ~mask & _not_null_like(values))
        return mask
    if kind == "between":
        _, _, low, high, negate = ast
        mask = cast(np.ndarray, (values >= low) & (values <= high))
        if negate:
            return cast(np.ndarray, ~mask & _not_null_like(values))
        return mask
    if kind == "isnull":
        _, _, negate = ast
        if np.issubdtype(values.dtype, np.floating):
            mask = np.isnan(values)
        else:
            mask = np.asarray([value is None for value in values], dtype=bool)
        return ~mask if negate else mask
    raise ValueError(f"Invalid AST node: {kind}")


__all__ = [
    "evaluate_where",
    "normalize_where_syntax",
    "parse_where_expression",
    "parse_where_literal",
    "tokenize_where_expression",
    "where_identifier_re",
    "where_columns_from_ast",
]
