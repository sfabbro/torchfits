"""Table interop: Arrow-native conversion to pandas, polars, DuckDB, parquet."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

# -- helpers shared with table.py ------------------------------------------------

from ..table import _require_pyarrow, _TABLE_IO_KEYS, read, scan, reader  # noqa: E402


# -- internal helpers (only used by the functions below) -------------------------


def _split_io_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    io_kwargs = {k: v for k, v in kwargs.items() if k in _TABLE_IO_KEYS}
    other_kwargs = {k: v for k, v in kwargs.items() if k not in _TABLE_IO_KEYS}
    return io_kwargs, other_kwargs


def _materialize_arrow_table(data: str | Any | Iterable[Any], **kwargs):
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
    **kwargs,
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

    if not stream:
        if hasattr(data, "read_next_batch"):
            table = pa.Table.from_batches(list(data))
        elif hasattr(data, "to_batches"):
            table = data
        else:
            table = pa.Table.from_batches(list(data))
        pq.write_table(
            table, where, compression=compression, row_group_size=row_group_size
        )
        return

    writer = None
    try:
        if hasattr(data, "read_next_batch"):
            while True:
                try:
                    batch = data.read_next_batch()
                except StopIteration:
                    break
                if writer is None:
                    writer = pq.ParquetWriter(
                        where, batch.schema, compression=compression
                    )
                writer.write_batch(batch, row_group_size=row_group_size)
        else:
            for batch in data:
                if writer is None:
                    writer = pq.ParquetWriter(
                        where, batch.schema, compression=compression
                    )
                writer.write_batch(batch, row_group_size=row_group_size)
    finally:
        if writer is not None:
            writer.close()


def to_pandas(
    data: str | Any | Iterable[Any],
    stream: bool = False,
    **kwargs,
):
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
    **kwargs,
):
    """
    Convert Arrow table data to polars DataFrame(s).

    Args:
        data: FITS file path, pyarrow.Table, or iterable of pyarrow.RecordBatch.
        stream: When True, return an iterator of polars DataFrames.
    """
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("polars is required for to_polars conversion") from exc

    if isinstance(data, str):
        io_kwargs, _ = _split_io_kwargs(kwargs)
        if stream:
            return (pl.from_arrow(batch) for batch in scan(data, **io_kwargs))
        return pl.from_arrow(read(data, **io_kwargs))

    if stream:
        return (pl.from_arrow(batch) for batch in data)

    return pl.from_arrow(data)


def to_polars_lazy(
    data: str | Any | Iterable[Any],
    **kwargs,
):
    """
    Convert table data into a Polars LazyFrame for complex expressions.
    """
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("polars is required for to_polars_lazy conversion") from exc

    table = _materialize_arrow_table(data, **kwargs)
    return pl.from_arrow(table).lazy()


def to_duckdb(
    data: str | Any | Iterable[Any],
    relation_name: str = "fits_table",
    connection: Any = None,
    **kwargs,
):
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
    **kwargs,
):
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
