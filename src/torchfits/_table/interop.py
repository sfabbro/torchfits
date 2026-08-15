"""Table interop: Arrow-native conversion to pandas, polars, DuckDB, parquet."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Optional

# -- helpers shared with table.py ------------------------------------------------

from .._table.utils import _require_pyarrow, _TABLE_IO_KEYS  # noqa: E402
from .._table.read import read, scan, reader  # noqa: E402


# -- internal helpers (only used by the functions below) -------------------------


def _split_io_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    io_kwargs = {k: v for k, v in kwargs.items() if k in _TABLE_IO_KEYS}
    other_kwargs = {k: v for k, v in kwargs.items() if k not in _TABLE_IO_KEYS}
    return io_kwargs, other_kwargs


def _materialize_arrow_table(data: str | Any | Iterable[Any], **kwargs: Any) -> Any:
    """Normalize path/reader/batches into a single pyarrow.Table."""
    pa = _require_pyarrow()

    if isinstance(data, str):
        io_kwargs, _ = _split_io_kwargs(kwargs)
        return read(data, **io_kwargs)

    if hasattr(data, "to_batches"):
        return data

    if hasattr(data, "read_next_batch"):
        return pa.Table.from_batches(list(data))

    if hasattr(pa, "RecordBatch") and isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data])

    return pa.Table.from_batches(list(data))


# -- public interop functions ----------------------------------------------------


def write_parquet(
    where: str,
    data: str | Any | Iterable[Any],
    *,
    stream: bool = False,
    compression: str = "zstd",
    row_group_size: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """
    Write Arrow-native table data to parquet.

    Args:
        where: Destination parquet file path.
        data: FITS file path, Arrow Table, RecordBatchReader, or iterable of RecordBatch.
        stream: Enable streaming parquet writes (bounded memory).
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow.parquet is required for parquet export") from exc

    pa = _require_pyarrow()

    if isinstance(data, str):
        if stream:
            data = reader(data, **kwargs)
        else:
            data = read(data, **kwargs)

    # Normalize to a concrete iterable after the str → reader/read path.
    data_iter: Any = data

    if not stream:
        if hasattr(data_iter, "read_next_batch"):
            table = pa.Table.from_batches(list(data_iter))
        elif hasattr(data_iter, "to_batches"):
            table = data_iter
        else:
            table = pa.Table.from_batches(list(data_iter))
        pq.write_table(
            table, where, compression=compression, row_group_size=row_group_size
        )
        return

    writer = None
    try:
        if hasattr(data_iter, "read_next_batch"):
            while True:
                try:
                    batch = data_iter.read_next_batch()
                except StopIteration:
                    break
                if writer is None:
                    writer = pq.ParquetWriter(
                        where, batch.schema, compression=compression
                    )
                writer.write_batch(batch, row_group_size=row_group_size)
        else:
            for batch in data_iter:
                if writer is None:
                    writer = pq.ParquetWriter(
                        where, batch.schema, compression=compression
                    )
                writer.write_batch(batch, row_group_size=row_group_size)
    finally:
        if writer is not None:
            writer.close()


def write_csv(
    where: str,
    data: str | Any | Iterable[Any],
    *,
    delimiter: str = ",",
    stream: bool = False,
    **kwargs: Any,
) -> None:
    """Write table data to CSV or TSV via PyArrow.

    Args:
        where: Destination path.
        data: FITS path, Arrow Table, reader, or batch iterable.
        delimiter: Field separator (``,`` for CSV, ``\\t`` for TSV).
        stream: Write batches without materializing the full table.
    """
    try:
        import pyarrow.csv as pacsv
    except ImportError as exc:
        raise ImportError("pyarrow.csv is required for CSV/TSV export") from exc

    write_options = pacsv.WriteOptions(delimiter=delimiter)

    if isinstance(data, str):
        data = reader(data, **kwargs) if stream else read(data, **kwargs)

    if not stream:
        table = _materialize_arrow_table(data, **kwargs)
        pacsv.write_csv(table, where, write_options=write_options)
        return

    data_iter: Any = data
    writer = None
    try:
        if hasattr(data_iter, "read_next_batch"):
            while True:
                try:
                    batch = data_iter.read_next_batch()
                except StopIteration:
                    break
                if writer is None:
                    writer = pacsv.CSVWriter(
                        where, batch.schema, write_options=write_options
                    )
                writer.write(batch)
        else:
            for batch in data_iter:
                if writer is None:
                    writer = pacsv.CSVWriter(
                        where, batch.schema, write_options=write_options
                    )
                writer.write(batch)
    finally:
        if writer is not None:
            writer.close()


def write_ipc(
    where: str,
    data: str | Any | Iterable[Any],
    *,
    stream: bool = False,
    compression: Optional[str] = "zstd",
    **kwargs: Any,
) -> None:
    """Write Arrow IPC / Feather V2 (``.arrow``) — native for Polars and Arrow.

    Args:
        where: Destination path (typically ``.arrow`` or ``.feather``).
        data: FITS path, Arrow Table, reader, or batch iterable.
        stream: Write batches without materializing the full table.
        compression: Feather/IPC compression (``zstd``, ``lz4``, or ``None``).
    """
    try:
        import pyarrow.feather as feather
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise ImportError(
            "pyarrow.feather/ipc is required for Arrow IPC export"
        ) from exc

    if isinstance(data, str):
        data = reader(data, **kwargs) if stream else read(data, **kwargs)

    if not stream:
        table = _materialize_arrow_table(data, **kwargs)
        feather.write_feather(table, where, compression=compression)
        return

    write_options = ipc.IpcWriteOptions(compression=compression)

    data_iter: Any = data
    writer = None
    try:
        if hasattr(data_iter, "read_next_batch"):
            while True:
                try:
                    batch = data_iter.read_next_batch()
                except StopIteration:
                    break
                if writer is None:
                    writer = ipc.new_file(where, batch.schema, options=write_options)
                writer.write_batch(batch)
        else:
            for batch in data_iter:
                if writer is None:
                    writer = ipc.new_file(where, batch.schema, options=write_options)
                writer.write_batch(batch)
    finally:
        if writer is not None:
            writer.close()


def to_pandas(
    data: str | Any | Iterable[Any],
    stream: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Convert Arrow table data to pandas.

    Args:
        data: FITS file path, pyarrow.Table, or iterable of pyarrow.RecordBatch.
        stream: When True, return an iterator of DataFrames.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for to_pandas conversion") from exc

    pa = _require_pyarrow()

    if isinstance(data, str):
        io_kwargs, pandas_kwargs = _split_io_kwargs(kwargs)
        if stream:
            return (
                pa.Table.from_batches([batch]).to_pandas(**pandas_kwargs)
                for batch in scan(data, **io_kwargs)
            )
        return read(data, **io_kwargs).to_pandas(**pandas_kwargs)

    if hasattr(data, "to_pandas"):
        return data.to_pandas(**kwargs)

    if stream:
        return (pa.Table.from_batches([batch]).to_pandas(**kwargs) for batch in data)

    frames = [pa.Table.from_batches([batch]).to_pandas(**kwargs) for batch in data]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def to_polars(
    data: str | Any | Iterable[Any],
    stream: bool = False,
    *,
    rechunk: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Convert Arrow table data to polars DataFrame(s).

    Args:
        data: FITS file path, pyarrow.Table, or iterable of pyarrow.RecordBatch.
        stream: When True, return an iterator of polars DataFrames.
        rechunk: When True (default False), force Polars to concatenate chunks
            into a single contiguous block.  Leaving this False avoids an
            unnecessary copy when the Arrow data is already a single chunk.
    """
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("polars is required for to_polars conversion") from exc

    if isinstance(data, str):
        io_kwargs, _ = _split_io_kwargs(kwargs)
        if stream:
            return (
                pl.from_arrow(batch, rechunk=rechunk)
                for batch in scan(data, **io_kwargs)
            )
        return pl.from_arrow(read(data, **io_kwargs), rechunk=rechunk)

    if stream:
        return (pl.from_arrow(batch, rechunk=rechunk) for batch in data)

    return pl.from_arrow(data, rechunk=rechunk)


def to_astropy(
    data: str | Any | Iterable[Any],
    **kwargs: Any,
) -> Any:
    """Convert FITS table path, Arrow Table, or record batches to an Astropy Table.

    Args:
        data: FITS file path, pyarrow.Table, or iterable of pyarrow.RecordBatch.
        **kwargs: Additional I/O keyword arguments passed to :func:`read` when
            data is a file path.

    Returns:
        astropy.table.Table: An Astropy Table containing the data.
    """
    import importlib

    try:
        astropy_table_mod = importlib.import_module("astropy.table")
        Table = astropy_table_mod.Table
    except ImportError as exc:
        raise ImportError("astropy is required for to_astropy conversion") from exc

    pa_table = _materialize_arrow_table(data, **kwargs)
    cols: dict[str, Any] = {}
    for name in pa_table.column_names:
        chunked_arr = pa_table[name]
        try:
            cols[name] = chunked_arr.to_numpy(zero_copy_only=False)
        except Exception:
            cols[name] = chunked_arr.to_pylist()
    return Table(cols)


def read_astropy(
    path: str,
    hdu: int | str = 1,
    **kwargs: Any,
) -> Any:
    """Read a FITS table directly into an Astropy Table.

    Args:
        path: FITS file path.
        hdu: Table HDU index or EXTNAME (default 1).
        **kwargs: Additional keyword arguments passed to :func:`read`
            (e.g. ``columns``, ``where``, ``mmap``, ``decode_bytes``).

    Returns:
        astropy.table.Table: An Astropy Table containing the read data.
    """
    return to_astropy(path, hdu=hdu, **kwargs)


def scan_polars(
    path: str,
    *,
    batch_size: int = 65536,
    rechunk: bool = False,
    **kwargs: Any,
) -> Iterator[Any]:
    """Stream FITS table data as Polars DataFrames, one batch at a time.

    This is a genuine streaming path: it yields one ``pl.DataFrame`` per
    internal batch without materializing the entire table.  Use this for large
    FITS tables that do not fit comfortably in memory or when you want to
    process chunks incrementally.

    Args:
        path: FITS file path.
        batch_size: Maximum number of rows per yielded DataFrame.
        rechunk: When True (default False), force Polars to concatenate chunks.
        **kwargs: Additional keyword arguments passed to :func:`scan`
            (e.g. ``hdu``, ``columns``, ``where``, ``mmap``, ``decode_bytes``).

    Yields:
        polars.DataFrame: One batch of rows as a Polars DataFrame.
    """
    from .._io_engine.paths import guard_fits_path

    # Fail closed on private URLs before the optional polars import.
    guard_fits_path(path)
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("polars is required for scan_polars conversion") from exc

    kwargs.setdefault("batch_size", batch_size)
    # Call scan() before creating the generator so remaining setup runs eagerly.
    batches = scan(path, **kwargs)
    return _scan_polars_iter(batches, rechunk=rechunk, pl=pl)


def _scan_polars_iter(
    batches: Iterator[Any], *, rechunk: bool, pl: Any
) -> Iterator[Any]:
    for batch in batches:
        yield pl.from_arrow(batch, rechunk=rechunk)


@dataclass
class FITSPolarsFrame:
    """A Polars DataFrame paired with FITS column/schema metadata.

    Returned by :func:`read_polars` so that FITS-specific metadata (TFORM,
    TUNIT, TDIM, TNULL, TSCAL, TZERO, HDU identity) is preserved alongside
    the Polars DataFrame.  The metadata is stored as a plain dict mapping
    column names to dicts of FITS keyword-value strings, plus a top-level
    ``table_meta`` dict for HDU-level information.

    Attribute access (``.height``, ``.columns``, ``.filter()``, etc.) delegates
    to the wrapped ``frame`` so the wrapper can be used like a regular
    ``pl.DataFrame`` in most contexts.
    """

    frame: Any
    field_meta: dict[str, dict[str, str]] = field(default_factory=dict)
    table_meta: dict[str, str] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.frame, name)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, key: Any) -> Any:
        return self.frame[key]

    def __repr__(self) -> str:
        meta_str = f", field_meta={self.field_meta!r}" if self.field_meta else ""
        if self.table_meta:
            meta_str += f", table_meta={self.table_meta!r}"
        return f"FITSPolarsFrame(frame={self.frame!r}{meta_str})"


def read_polars(
    path: str,
    *,
    rechunk: bool = False,
    **kwargs: Any,
) -> FITSPolarsFrame:
    """Read a FITS table directly as a Polars DataFrame with FITS metadata.

    This is a convenience wrapper that calls :func:`read` with
    ``include_fits_metadata=True``, extracts the FITS field and table metadata
    from the Arrow schema, converts the table to Polars via
    ``pl.from_arrow(rechunk=rechunk)``, and returns an
    :class:`FITSPolarsFrame` containing both the DataFrame and the metadata.

    Unlike :func:`to_polars`, this preserves FITS metadata (TFORM, TUNIT,
    TDIM, TNULL, TSCAL, TZERO, HDU identity) that would otherwise be lost
    when Polars consumes the Arrow table.

    Args:
        path: FITS file path.
        rechunk: When True (default False), force Polars to concatenate chunks.
        **kwargs: Additional I/O keyword arguments passed to :func:`read`
            (e.g. ``hdu``, ``columns``, ``where``, ``mmap``, ``decode_bytes``,
            ``encoding``, ``strip``, ``apply_fits_nulls``, ``backend``).

    Returns:
        FITSPolarsFrame: A wrapper around a ``polars.DataFrame`` with
        ``.field_meta`` and ``.table_meta`` attributes.  The wrapper delegates
        attribute access to the DataFrame, so it can be used like a regular
        Polars DataFrame in most contexts.
    """
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("polars is required for read_polars conversion") from exc

    # Force metadata inclusion so we can extract it before Polars conversion.
    kwargs.setdefault("include_fits_metadata", True)
    arrow_table = read(path, **kwargs)

    # Extract FITS metadata from the Arrow schema before conversion.
    field_meta, table_meta = _fits_meta_from_arrow_schema(arrow_table.schema)

    df = pl.from_arrow(arrow_table, rechunk=rechunk)
    return FITSPolarsFrame(frame=df, field_meta=field_meta, table_meta=table_meta)


def _fits_meta_from_arrow_schema(
    schema: Any,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Decode Arrow schema/field metadata into FITS string maps."""
    table_meta: dict[str, str] = {}
    field_meta: dict[str, dict[str, str]] = {}

    md = schema.metadata
    if md:
        decode = _decode_meta_bytes
        table_meta = {decode(k): decode(v) for k, v in md.items()}

    names = schema.names
    for i, name in enumerate(names):
        fmd = schema.field(i).metadata
        if not fmd:
            continue
        entry = {_decode_meta_bytes(k): _decode_meta_bytes(v) for k, v in fmd.items()}
        if entry:
            field_meta[name] = entry
    return field_meta, table_meta


def _decode_meta_bytes(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def to_duckdb(
    data: str | Any | Iterable[Any],
    relation_name: str = "fits_table",
    connection: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Register table data in DuckDB and return a relation.

    This is intended for SQL-style joins/group-bys/windows while keeping torchfits
    focused on FITS-native I/O and conversion.
    """
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError("duckdb is required for to_duckdb conversion") from exc

    if not isinstance(relation_name, str) or not relation_name:
        raise ValueError("relation_name must be a non-empty string")

    arrow_table = _materialize_arrow_table(data, **kwargs)
    con = connection if connection is not None else duckdb.connect()
    con.register(relation_name, arrow_table)
    return con.table(relation_name)


def duckdb_query(
    data: str | Any | Iterable[Any],
    query: str,
    relation_name: str = "fits_table",
    connection: Any = None,
    return_arrow: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Execute a DuckDB SQL query over table data.
    """
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError("duckdb is required for duckdb_query") from exc

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty SQL string")

    con = connection if connection is not None else duckdb.connect()
    _ = to_duckdb(
        data,
        relation_name=relation_name,
        connection=con,
        **kwargs,
    )
    # Prevent SQL injection by strictly enforcing exactly one SELECT or EXPLAIN statement
    statements = duckdb.extract_statements(query)
    if len(statements) != 1:
        raise ValueError("query must contain exactly one SQL statement")

    stmt_type = statements[0].type
    if stmt_type not in {duckdb.StatementType.SELECT, duckdb.StatementType.EXPLAIN}:
        raise ValueError(
            f"query must be a SELECT or EXPLAIN statement, got {stmt_type}"
        )

    result = con.sql(query)
    if return_arrow:
        return result.arrow()
    return result
