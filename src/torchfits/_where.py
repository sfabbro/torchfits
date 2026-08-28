from functools import lru_cache
import re
import ast
from typing import Any, List, Tuple

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
    """Translate C-style logical operators to SQL/Python style before parsing.

    Rewrites apply only *outside* quoted literals, so values such as
    ``'AT&T'`` or ``'x||y'`` survive verbatim instead of being silently
    rewritten into ``AND``/``OR``. FITS-style doubled-quote escapes inside
    single-quoted literals (``'O''NEIL'``) are converted to Python
    backslash escapes so ``ast.parse`` yields the intended value.
    """
    out: list[str] = []
    i = 0
    n = len(where)
    while i < n:
        ch = where[i]
        if ch in ("'", '"'):
            quote = ch
            i += 1
            buf: List[str] = []
            closed = False
            while i < n:
                cur = where[i]
                if quote == "'" and cur == "'" and i + 1 < n and where[i + 1] == "'":
                    # FITS escaped quote -> Python escape sequence.
                    buf.append("\\'")
                    i += 2
                    continue
                if cur == "\\" and i + 1 < n:
                    buf.append(cur)
                    buf.append(where[i + 1])
                    i += 2
                    continue
                if cur == quote:
                    closed = True
                    break
                buf.append(cur)
                i += 1
            if not closed:
                raise ValueError("Unterminated quoted literal in where expression")
            i += 1  # consume closing quote
            content = "".join(buf)
            if quote == '"':
                # Re-emit as a safe double-quoted literal for ast.parse.
                out.append('"')
                out.append(content.replace('"', '\\"'))
                out.append('"')
            else:
                out.append("'")
                out.append(content)
                out.append("'")
            continue
        two = where[i : i + 2]
        if two == "&&":
            out.append(" AND ")
            i += 2
            continue
        if two == "||":
            out.append(" OR ")
            i += 2
            continue
        if ch == "&":
            out.append(" AND ")
            i += 1
            continue
        if ch == "|":
            out.append(" OR ")
            i += 1
            continue
        if ch == "~":
            prev = where[i - 1] if i > 0 else ""
            nxt = where[i + 1] if i + 1 < n else ""

            def _word(c: str) -> bool:
                return c.isalnum() or c == "_"

            if not _word(prev) and not _word(nxt):
                out.append(" NOT ")
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _normalize_logical_operators(where: str) -> str:
    parts: list[str] = re.split(r"('[^']*'|\"[^\"]*\")", where)
    for i in range(0, len(parts), 2):
        s = parts[i]
        s = re.sub(r"\bAND\b", "and", s, flags=re.IGNORECASE)
        s = re.sub(r"\bOR\b", "or", s, flags=re.IGNORECASE)
        s = re.sub(r"\bNOT\b", "not", s, flags=re.IGNORECASE)
        s = re.sub(r"\bIN\b", "in", s, flags=re.IGNORECASE)
        parts[i] = s
    return "".join(parts)


# Numeric literal or bare-word pattern used for BETWEEN boundary values.
# Prefer explicit numeric/word matching over bare \S+ to avoid accidentally
# capturing comparison operators or parentheses in pathological expressions.
_BOUNDARY_VALUE = r"('[^']*'|\"[^\"]*\"|[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\w+)"


def _normalize_between(where: str) -> str:
    where = re.sub(
        r"\b(\w+)\s+NOT\s+BETWEEN\s+"
        + _BOUNDARY_VALUE
        + r"\s+AND\s+"
        + _BOUNDARY_VALUE,
        r"not_between(\1, \2, \3)",
        where,
        flags=re.IGNORECASE,
    )
    where = re.sub(
        r"\b(\w+)\s+BETWEEN\s+" + _BOUNDARY_VALUE + r"\s+AND\s+" + _BOUNDARY_VALUE,
        r"between(\1, \2, \3)",
        where,
        flags=re.IGNORECASE,
    )
    return where


def _normalize_nulls(where: str) -> str:
    where = re.sub(
        r"\b(\w+)\s+IS\s+NOT\s+NULL\b", r"isnotnull(\1)", where, flags=re.IGNORECASE
    )
    where = re.sub(
        r"\b(\w+)\s+NOT\s+NULL\b", r"isnotnull(\1)", where, flags=re.IGNORECASE
    )
    where = re.sub(r"\b(\w+)\s+IS\s+NULL\b", r"isnull(\1)", where, flags=re.IGNORECASE)
    return where


def _get_constant_val(node: Any) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str) and node.value.upper() in {"NULL", "NONE"}:
            return None
        return node.value
    if isinstance(node, ast.Name):
        if node.id.upper() in {"NULL", "NONE"}:
            return None
        return node.id
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _get_constant_val(node.operand)
        if isinstance(val, (int, float)):
            return -val
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _get_constant_val(node.operand)
    raise ValueError("Expected a constant value")


def _extract_literals(node: Any) -> list[Any]:
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_get_constant_val(el) for el in node.elts]
    return [_get_constant_val(node)]


def _to_custom_ast(node: Any) -> tuple[Any, ...]:
    if isinstance(node, ast.Expression):
        return _to_custom_ast(node.body)

    if isinstance(node, ast.BoolOp):
        op_name = "and" if isinstance(node.op, ast.And) else "or"
        current = _to_custom_ast(node.values[0])
        for val in node.values[1:]:
            current = (op_name, current, _to_custom_ast(val))
        return current

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return ("not", _to_custom_ast(node.operand))

    if isinstance(node, ast.Compare):
        left = node.left
        if not isinstance(left, ast.Name):
            raise ValueError(
                "where expects a column identifier before comparison operator"
            )
        col_name = left.id

        if len(node.ops) != 1:
            raise ValueError("Unsupported complex comparison")

        op = node.ops[0]
        comparator = node.comparators[0]

        if isinstance(op, ast.In):
            literals = _extract_literals(comparator)
            return ("in", col_name, literals, False)

        if isinstance(op, ast.NotIn):
            literals = _extract_literals(comparator)
            return ("in", col_name, literals, True)

        op_map = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Lt: "<",
            ast.LtE: "<=",
        }
        if type(op) not in op_map:
            raise ValueError(f"Unsupported operator: {op}")

        literal = _get_constant_val(comparator)
        return ("cmp", col_name, op_map[type(op)], literal)

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in {"_between", "_not_between", "between", "not_between"}:
                if len(node.args) != 3 or not isinstance(node.args[0], ast.Name):
                    raise ValueError("Invalid between call")
                col_name = node.args[0].id
                low = _get_constant_val(node.args[1])
                high = _get_constant_val(node.args[2])
                negate = func_name in {"_not_between", "not_between"}
                return ("between", col_name, low, high, negate)

            if func_name in {"_isnull", "_isnotnull", "isnull", "isnotnull"}:
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Name):
                    raise ValueError("Invalid null check call")
                col_name = node.args[0].id
                negate = func_name in {"_isnotnull", "isnotnull"}
                return ("isnull", col_name, negate)
    raise ValueError(
        "where expects a comparison operator or IN/BETWEEN/IS NULL variants after column identifier"
    )


@lru_cache(maxsize=1024)
def _parse_where_expression(where: str) -> tuple[Any, ...]:
    if not isinstance(where, str) or not where.strip():
        raise ValueError("where must be a non-empty string expression")
    try:
        where_normalized = _normalize_where_syntax(where)
        where_normalized = _normalize_logical_operators(where_normalized)
        where_normalized = _normalize_between(where_normalized)
        where_normalized = _normalize_nulls(where_normalized)

        node = ast.parse(where_normalized.strip(), mode="eval")
        return _to_custom_ast(node)
    except SyntaxError as e:
        if where.count("(") != where.count(")"):
            raise ValueError("Unbalanced parentheses in where expression")
        if "end of" in str(e) or "EOF" in str(e) or "unexpected EOF" in str(e):
            raise ValueError("Unexpected end of where expression")
        raise ValueError("Unexpected trailing tokens in where expression")


def _where_columns_from_ast(ast: tuple[Any, ...]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()

    def _visit(node: tuple[Any, ...]) -> None:
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


# Public aliases used by _table/read.py (WHERE predicate parsing + column extraction).
parse_where_expression = _parse_where_expression
where_columns_from_ast = _where_columns_from_ast
