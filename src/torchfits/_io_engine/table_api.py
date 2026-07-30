"""Table-shaped wrappers over unified FITS reads."""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch

from .hdu_api import _resolve_hdu_index, autodetect_hdu


def _resolve_mmap(mmap: Union[bool, str]) -> bool:
    if isinstance(mmap, bool):
        return mmap
    # "auto" → prefer mmap for tables (CFITSIO column mmap path).
    return True


def _move_table_dict(data: dict[str, Any], device: str) -> dict[str, Any]:
    if device == "cpu":
        return data
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device=device)
        else:
            out[key] = value
    return out


def _apply_row_window(
    data: dict[str, Any], start_row: int, num_rows: int
) -> dict[str, Any]:
    """Apply 1-based FITS row window after a full-table filter."""
    if int(start_row) <= 1 and int(num_rows) < 0:
        return data
    start0 = max(0, int(start_row) - 1)
    stop = None if int(num_rows) < 0 else start0 + int(num_rows)
    out: dict[str, Any] = {}
    for key, value in data.items():
        out[key] = value[start0:stop] if isinstance(value, torch.Tensor) else value
    return out


def _thin_read_table_torch(
    path: str,
    *,
    hdu: int,
    columns: Optional[list[str]],
    start_row: int,
    num_rows: int,
    device: str,
    mmap: Union[bool, str],
) -> dict[str, Any]:
    """CFITSIO → torch dict without going through read_unified image probes."""
    import torchfits._C as cpp

    col_names = list(columns) if columns is not None else []
    use_mmap = _resolve_mmap(mmap)
    data = cpp.read_fits_table_rows(
        path, int(hdu), col_names, int(start_row), int(num_rows), bool(use_mmap)
    )
    return _move_table_dict(dict(data), device)


def _thin_read_table_filtered(
    path: str,
    *,
    hdu: int,
    columns: Optional[list[str]],
    where: str,
    device: str,
    compile_predicates: Callable[[str], Any],
) -> dict[str, Any] | None:
    """Fused project+predicate via cpp.read_fits_table_filtered (gather path).

    Prefer the project+mask path in :func:`read_table` for typical keep rates;
    this remains as a fallback when the thin full-column read fails.
    """
    import torchfits._C as cpp

    if not hasattr(cpp, "read_fits_table_filtered"):
        return None
    filters = compile_predicates(where)
    if filters is None:
        return None
    target_cols = list(columns) if columns is not None else []
    if not target_cols:
        # Filtered binding needs explicit columns; pull colnames skinny.
        from .hdu_api import read_colnames

        target_cols = read_colnames(path, hdu=hdu)
    try:
        data = cpp.read_fits_table_filtered(path, int(hdu), target_cols, list(filters))
    except Exception:
        return None
    return _move_table_dict(dict(data), device)


def read_table(
    read_func: Callable[..., Any],
    path: str,
    hdu: Union[int, str] = 1,
    columns: Optional[list[str]] = None,
    start_row: int = 1,
    num_rows: int = -1,
    device: str = "cpu",
    mmap: Union[bool, str] = "auto",
    cache_capacity: int = 10,
    handle_cache_capacity: int = 16,
    fast_header: bool = True,
    return_header: bool = False,
    where: str | None = None,
) -> Any:
    """Read a table HDU as a dictionary of tensors/lists.

    Thin path (default): C++ ``read_fits_table_rows`` / filtered binding.
    Falls back to ``read_func`` (``torchfits.read``) when header is requested
    or the thin path fails. ``hdu`` may be an index or EXTNAME.
    """
    _ = cache_capacity, handle_cache_capacity, fast_header, autodetect_hdu
    if hdu is None or (isinstance(hdu, str) and hdu.strip().lower() == "auto"):
        raise ValueError(
            "hdu must be a non-negative integer or EXTNAME string "
            "(not None/'auto'); pass an explicit table HDU"
        )
    if isinstance(hdu, int):
        if hdu < 0:
            raise ValueError("hdu must be a non-negative integer")
    elif isinstance(hdu, str):
        hdu = _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    else:
        raise TypeError(f"hdu must be int or str, got {type(hdu)!r}")

    if where is not None and str(where).strip():
        from torchfits._table.read import _compile_where_to_simple_predicates

        predicates = _compile_where_to_simple_predicates(str(where))
        if predicates is None:
            raise ValueError(f"Unsupported where expression for read_torch: {where!r}")

        # Prefer project-all-needed-cols + torch mask over ``read_fits_table_filtered``.
        # Filtered gather wins only when keep-rate is tiny *and* row assembly is
        # heavy; for narrow/dense predicates (the Round-3 deficit cluster) mask
        # is several× faster and also beats Astropy's numpy mask path.
        pred_cols = [col for col, _op, _lit in predicates]
        read_cols = list(columns) if columns is not None else []
        for name in pred_cols:
            if name not in read_cols:
                read_cols.append(name)
        try:
            data = _thin_read_table_torch(
                path,
                hdu=hdu,
                columns=read_cols or None,
                start_row=1,
                num_rows=-1,
                device="cpu",
                mmap=mmap,
            )
        except Exception:
            data = None

        if data is None:
            pushed = _thin_read_table_filtered(
                path,
                hdu=hdu,
                columns=columns,
                where=str(where),
                device=device,
                compile_predicates=_compile_where_to_simple_predicates,
            )
            if pushed is None:
                raise RuntimeError(
                    f"Failed to apply where={where!r} on table HDU {hdu!r}"
                )
            data = _apply_row_window(pushed, start_row, num_rows)
            if return_header:
                import torchfits

                return data, torchfits.read_header(path, hdu=hdu)
            return data

        mask: torch.Tensor | None = None
        for col, op, lit in predicates:
            values = data[col]
            if not isinstance(values, torch.Tensor):
                values = torch.as_tensor(values)
            if op == ">":
                part = values > lit
            elif op == ">=":
                part = values >= lit
            elif op == "<":
                part = values < lit
            elif op == "<=":
                part = values <= lit
            elif op == "==":
                part = values == lit
            elif op == "!=":
                part = values != lit
            else:
                raise ValueError(f"Unsupported where operator {op!r}")
            mask = part if mask is None else (mask & part)
        assert mask is not None
        keep_cols = list(columns) if columns is not None else list(data.keys())
        filtered = {
            k: (v[mask] if isinstance(v, torch.Tensor) else v)
            for k, v in data.items()
            if k in keep_cols
        }
        data = _apply_row_window(
            _move_table_dict(filtered, device), start_row, num_rows
        )
        if return_header:
            import torchfits

            return data, torchfits.read_header(path, hdu=hdu)
        return data

    if not return_header:
        try:
            data = _thin_read_table_torch(
                path,
                hdu=hdu,
                columns=columns,
                start_row=start_row,
                num_rows=num_rows,
                device=device,
                mmap=mmap,
            )
            if isinstance(data, torch.Tensor):
                raise ValueError(
                    f"HDU {hdu!r} is an image HDU. Use read_tensor(...) or read(...)."
                )
            return data
        except ValueError:
            raise
        except Exception:
            pass

    out = read_func(
        path=path,
        hdu=hdu,
        mode="table",
        device=device,
        mmap=mmap,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        cache_capacity=cache_capacity,
        handle_cache_capacity=handle_cache_capacity,
        fast_header=fast_header,
        return_header=return_header,
    )
    data = out[0] if return_header else out
    if isinstance(data, torch.Tensor):
        raise ValueError(
            f"HDU {hdu!r} is an image HDU. Use read_tensor(...) or read(...)."
        )
    return out
