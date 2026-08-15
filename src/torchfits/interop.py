from typing import TYPE_CHECKING, Any, Dict, cast

import torch

from ._string_decode import decode_byte_tensor as _decode_byte_tensor
from ._tensor_buffer import tensor_to_arrow_array as _tensor_to_arrow_array

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    import pyarrow as pa


# -- public interop functions ----------------------------------------------------


def to_pandas(
    data: Dict[str, Any],
    decode_bytes: bool = False,
    encoding: str = "ascii",
    strip: bool = True,
    vla_policy: str = "object",
) -> "pd.DataFrame":
    """
    Convert a dictionary of PyTorch tensors to a Pandas DataFrame.

    Routes through :func:`to_arrow` then ``pyarrow.Table.to_pandas`` so no
    numpy hop is required.  Zero-copy is used where possible for numeric
    columns.

    Args:
        data: Dictionary mapping column names to PyTorch tensors or lists of tensors (VLA).

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the data.
    """
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        raise ImportError("Pandas is required for to_pandas conversion.") from None

    # Map pandas VLA policies to Arrow VLA policies.
    if vla_policy == "object":
        arrow_vla = "list"
    elif vla_policy == "drop":
        arrow_vla = "drop"
    else:
        raise ValueError("vla_policy must be 'object' or 'drop'")
    arrow_table = to_arrow(
        data,
        decode_bytes=decode_bytes,
        encoding=encoding,
        strip=strip,
        vla_policy=arrow_vla,
    )
    return arrow_table.to_pandas()


def to_polars(
    data: Dict[str, Any],
    decode_bytes: bool = False,
    encoding: str = "ascii",
    strip: bool = True,
    vla_policy: str = "list",
    *,
    rechunk: bool = False,
) -> "pl.DataFrame":
    """
    Convert a dictionary of PyTorch tensors to a Polars DataFrame (via PyArrow).

    Same arguments as :func:`to_arrow`; uses :func:`polars.from_arrow` on the
    intermediate table so the result stays Arrow-backed.

    Args:
        rechunk: When True (default False), force Polars to concatenate chunks
            into a single contiguous block.  Leaving False avoids an unnecessary
            copy when the Arrow data is already a single chunk.
    """
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("Polars is required for to_polars conversion.") from exc

    return cast(
        "pl.DataFrame",
        pl.from_arrow(
            to_arrow(
                data,
                decode_bytes=decode_bytes,
                encoding=encoding,
                strip=strip,
                vla_policy=vla_policy,
            ),
            rechunk=rechunk,
        ),
    )


def to_astropy(
    data: Dict[str, Any],
    decode_bytes: bool = False,
    encoding: str = "ascii",
    strip: bool = True,
    vla_policy: str = "list",
) -> Any:
    """Convert a dictionary of PyTorch tensors to an Astropy Table."""
    import importlib

    try:
        astropy_table_mod = importlib.import_module("astropy.table")
        Table = astropy_table_mod.Table
    except ImportError:
        raise ImportError("Astropy is required for to_astropy conversion.") from None

    arrow_table = to_arrow(
        data,
        decode_bytes=decode_bytes,
        encoding=encoding,
        strip=strip,
        vla_policy=vla_policy,
    )
    cols: dict[str, Any] = {}
    for name in arrow_table.column_names:
        chunked_arr = arrow_table[name]
        try:
            cols[name] = chunked_arr.to_numpy(zero_copy_only=False)
        except Exception:
            cols[name] = chunked_arr.to_pylist()
    return Table(cols)


def to_arrow(
    data: Dict[str, Any],
    decode_bytes: bool = False,
    encoding: str = "ascii",
    strip: bool = True,
    vla_policy: str = "list",
) -> "pa.Table":
    """
    Convert a dictionary of PyTorch tensors to a PyArrow Table.

    Uses ``bytes(tensor.untyped_storage())`` + ``pa.Array.from_buffers`` for
    numeric columns (one copy, same cost as the previous numpy path) and
    pure-Python byte decoding for string columns.  No numpy dependency.

    Args:
        data: Dictionary mapping column names to PyTorch tensors or lists of tensors (VLA).

    Returns:
        pa.Table: A PyArrow Table containing the data.
    """
    try:
        import pyarrow as pa
    except ImportError:
        raise ImportError("PyArrow is required for to_arrow conversion.") from None

    arrays = []
    names = []

    for key, value in data.items():
        names.append(key)
        if isinstance(value, torch.Tensor):
            if decode_bytes and value.dtype == torch.uint8 and value.dim() == 2:
                # Decode fixed-width byte column to strings (pure Python).
                decoded = _decode_byte_tensor(value, encoding=encoding, strip=strip)
                arrays.append(pa.array(decoded))
            else:
                arrays.append(_tensor_to_arrow_array(value, pa))
        elif isinstance(value, list):
            # Handle VLA (list of tensors).
            if vla_policy == "list":
                converted_list = [
                    t.tolist() if isinstance(t, torch.Tensor) else t for t in value
                ]
                arrays.append(pa.array(converted_list))
            elif vla_policy == "drop":
                names.pop()
            else:
                raise ValueError("vla_policy must be 'list' or 'drop'")
        else:
            arrays.append(pa.array(value))

    return pa.Table.from_arrays(arrays, names=names)
