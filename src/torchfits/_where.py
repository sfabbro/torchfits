import re
from typing import Any, Optional, List, Tuple
from functools import lru_cache

_WHERE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_where_literal(raw: str) -> Any:
    token = raw.strip()
    if not token:
        raise ValueError("where literal cannot be empty")

    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        quote = token[0]
        inner = token[1:-1]
        return inner.replace(f"\\{quote}", quote)

    token_lower = token.lower()
    if token_lower == "true":
        return True
    if token_lower == "false":
        return False
    if token_lower in {"none", "null"}:
        return None

    if re.fullmatch(r"[+-]?\d+", token):
        try:
            return int(token)
        except Exception:
            pass

    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token):
        try:
            return float(token)
        except Exception:
            pass

    # Bare-word strings are accepted (e.g. where="NAME == STAR_A").
    return token


def _tokenize_where_expression(where: str) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    i = 0
    n = len(where)
    while i < n:
        ch = where[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append(("LPAREN", ch))
            i += 1
            continue
        if ch == ")":
            tokens.append(("RPAREN", ch))
            i += 1
            continue
        if ch == ",":
            tokens.append(("COMMA", ch))
            i += 1
            continue

        if i + 1 < n:
            op2 = where[i : i + 2]
            if op2 in {"==", "!=", ">=", "<="}:
                tokens.append(("OP", op2))
                i += 2
                continue
        if ch in {">", "<"}:
            tokens.append(("OP", ch))
            i += 1
            continue

        if ch in {"'", '"'}:
            quote = ch
            i += 1
            buf: List[str] = []
            while i < n:
                cur = where[i]
                if cur == "\\" and i + 1 < n:
                    buf.append(where[i + 1])
                    i += 2
                    continue
                if cur == quote:
                    break
                buf.append(cur)
                i += 1
            if i >= n or where[i] != quote:
                raise ValueError("Unterminated quoted literal in where expression")
            i += 1
            tokens.append(("LITERAL", "".join(buf)))
            continue

        start = i
        while i < n:
            cur = where[i]
            if cur.isspace() or cur in {"(", ")", ",", ">", "<", "!", "="}:
                break
            i += 1
        token = where[start:i]
        if not token:
            raise ValueError(
                f"Unexpected token in where expression near position {start}"
            )
        tokens.append(("WORD", token))

    return tokens


def _normalize_where_syntax(where: str) -> str:
    """Translate C-style logical operators to SQL-style before parsing."""
    result = where.replace("&&", " AND ").replace("||", " OR ")
    result = re.sub(r"(?<!\w)~(?!\w)", " NOT ", result)
    result = re.sub(r"(?<![!=<>])&(?!&)", " AND ", result)
    result = re.sub(r"(?<!\|)\|(?!\|)", " OR ", result)
    return result


@lru_cache(maxsize=1024)
def _parse_where_expression(where: str):
    if not isinstance(where, str) or not where.strip():
        raise ValueError("where must be a non-empty string expression")
    where = _normalize_where_syntax(where)

    tokens = _tokenize_where_expression(where)
    if not tokens:
        raise ValueError("where must be a non-empty string expression")

    idx = 0

    def _peek() -> Optional[Tuple[str, str]]:
        return tokens[idx] if idx < len(tokens) else None

    def _consume() -> Tuple[str, str]:
        nonlocal idx
        if idx >= len(tokens):
            raise ValueError("Unexpected end of where expression")
        out = tokens[idx]
        idx += 1
        return out

    def _consume_logic(expected: str) -> None:
        tok = _peek()
        if tok is None or tok[0] != "WORD" or tok[1].upper() != expected:
            raise ValueError(f"Expected '{expected}' in where expression")
        _consume()

    def _parse_literal_token(tok: Tuple[str, str]) -> Any:
        if tok[0] == "LITERAL":
            return tok[1]
        if tok[0] == "WORD":
            return _parse_where_literal(tok[1])
        raise ValueError("where expects a literal value")

    def _parse_literal_list() -> List[Any]:
        head = _consume()
        if head[0] != "LPAREN":
            raise ValueError("where IN expects '(' after IN")
        literals: List[Any] = []
        while True:
            tok = _peek()
            if tok is None:
                raise ValueError("Unexpected end of where expression in IN list")
            if tok[0] == "RPAREN":
                _consume()
                break
            if literals:
                sep = _consume()
                if sep[0] != "COMMA":
                    raise ValueError("where IN expects ',' between list literals")
            tok = _consume()
            literals.append(_parse_literal_token(tok))
        return literals

    def _parse_comparison():
        lhs = _consume()
        if lhs[0] != "WORD" or _WHERE_IDENT_RE.fullmatch(lhs[1]) is None:
            raise ValueError(
                "where expects a column identifier before comparison operator"
            )
        op_tok = _peek()
        if op_tok is None:
            raise ValueError(
                "where expects a comparison operator after column identifier"
            )

        if op_tok[0] == "OP":
            _consume()
            rhs = _consume()
            literal = _parse_literal_token(rhs)
            return ("cmp", lhs[1], op_tok[1], literal)

        if op_tok[0] == "WORD" and op_tok[1].upper() == "IN":
            _consume_logic("IN")
            return ("in", lhs[1], _parse_literal_list(), False)

        if op_tok[0] == "WORD" and op_tok[1].upper() == "BETWEEN":
            _consume_logic("BETWEEN")
            low = _parse_literal_token(_consume())
            _consume_logic("AND")
            high = _parse_literal_token(_consume())
            return ("between", lhs[1], low, high, False)

        if op_tok[0] == "WORD" and op_tok[1].upper() == "IS":
            _consume_logic("IS")
            next_tok = _peek()
            negate = False
            if (
                next_tok is not None
                and next_tok[0] == "WORD"
                and next_tok[1].upper() == "NOT"
            ):
                _consume_logic("NOT")
                negate = True
            _consume_logic("NULL")
            return ("isnull", lhs[1], negate)

        if op_tok[0] == "WORD" and op_tok[1].upper() == "NOT":
            _consume_logic("NOT")
            next_tok = _peek()
            if (
                next_tok is not None
                and next_tok[0] == "WORD"
                and next_tok[1].upper() == "IN"
            ):
                _consume_logic("IN")
                return ("in", lhs[1], _parse_literal_list(), True)
            if (
                next_tok is not None
                and next_tok[0] == "WORD"
                and next_tok[1].upper() == "BETWEEN"
            ):
                _consume_logic("BETWEEN")
                low = _parse_literal_token(_consume())
                _consume_logic("AND")
                high = _parse_literal_token(_consume())
                return ("between", lhs[1], low, high, True)
            if (
                next_tok is not None
                and next_tok[0] == "WORD"
                and next_tok[1].upper() == "NULL"
            ):
                _consume_logic("NULL")
                return ("isnull", lhs[1], True)
            raise ValueError("where expects IN/BETWEEN/NULL after NOT")

        raise ValueError(
            "where expects a comparison operator or IN/BETWEEN/IS NULL variants after column identifier"
        )

    def _parse_primary():
        tok = _peek()
        if tok is None:
            raise ValueError("Unexpected end of where expression")
        if tok[0] == "LPAREN":
            _consume()
            node = _parse_or()
            tail = _consume()
            if tail[0] != "RPAREN":
                raise ValueError("Unbalanced parentheses in where expression")
            return node
        return _parse_comparison()

    def _parse_not():
        tok = _peek()
        if tok is not None and tok[0] == "WORD" and tok[1].upper() == "NOT":
            _consume_logic("NOT")
            return ("not", _parse_not())
        return _parse_primary()

    def _parse_and():
        node = _parse_not()
        while True:
            tok = _peek()
            if tok is not None and tok[0] == "WORD" and tok[1].upper() == "AND":
                _consume_logic("AND")
                rhs = _parse_not()
                node = ("and", node, rhs)
                continue
            return node

    def _parse_or():
        node = _parse_and()
        while True:
            tok = _peek()
            if tok is not None and tok[0] == "WORD" and tok[1].upper() == "OR":
                _consume_logic("OR")
                rhs = _parse_and()
                node = ("or", node, rhs)
                continue
            return node

    ast = _parse_or()
    if idx != len(tokens):
        raise ValueError("Unexpected trailing tokens in where expression")
    return ast


def _where_columns_from_ast(ast) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()

    def _visit(node) -> None:
        kind = node[0]
        if kind in {"cmp", "in", "between", "isnull"}:
            name = node[1]
            if name not in seen:
                seen.add(name)
                out.append(name)
        elif kind == "and" or kind == "or":
            _visit(node[1])
            _visit(node[2])
        elif kind == "not":
            _visit(node[1])
        else:
            raise ValueError("Invalid where AST")

    _visit(ast)
    return out


def evaluate_where(ast: Tuple, data_map: dict) -> Any:
    """Evaluate a where expression AST against a data map of NumPy arrays."""
    import numpy as np

    kind = ast[0]

    if kind == "cmp":
        _, col, op, literal = ast
        if col not in data_map:
            raise ValueError(f"Unknown column: {col}")
        val = data_map[col]

        if literal is None:
            if op == "==":
                return np.array([v is None for v in val])
            if op == "!=":
                return np.array([v is not None for v in val])
            raise ValueError("NULL comparisons only support == and !=")

        if op == "==":
            return val == literal
        if op == "!=":
            return val != literal
        if op == ">":
            return val > literal
        if op == ">=":
            return val >= literal
        if op == "<":
            return val < literal
        if op == "<=":
            return val <= literal
        raise ValueError(f"Unsupported operator: {op}")

    if kind == "and":
        return evaluate_where(ast[1], data_map) & evaluate_where(ast[2], data_map)

    if kind == "or":
        return evaluate_where(ast[1], data_map) | evaluate_where(ast[2], data_map)

    if kind == "not":
        return ~evaluate_where(ast[1], data_map)

    if kind == "in":
        _, col, literals, negate = ast
        if col not in data_map:
            raise ValueError(f"Unknown column: {col}")
        val = data_map[col]
        # NumPy's isin handles lists of literals
        mask = np.isin(val, literals)
        return ~mask if negate else mask

    if kind == "between":
        _, col, low, high, negate = ast
        if col not in data_map:
            raise ValueError(f"Unknown column: {col}")
        val = data_map[col]
        mask = (val >= low) & (val <= high)
        return ~mask if negate else mask

    if kind == "isnull":
        _, col, negate = ast
        if col not in data_map:
            raise ValueError(f"Unknown column: {col}")
        val = data_map[col]
        # For numpy arrays, check for None or NaN depending on dtype
        if np.issubdtype(val.dtype, np.floating):
            mask = np.isnan(val)
        else:
            # For object arrays or other types, check for None
            mask = np.array([v is None for v in val])
        return ~mask if negate else mask

    raise ValueError(f"Invalid AST node: {kind}")
