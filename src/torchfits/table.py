"""Arrow-native table I/O helpers.

FITS tables are dataframes on disk. The namespace is ``table`` (FITS name);
destinations are Arrow (``read`` / ``read_arrow``), tensor columns
(``read_torch``), or Polars (``read_polars``).
"""

from __future__ import annotations

from ._table.interop import (
    FITSPolarsFrame,
    duckdb_query,
    read_astropy,
    read_polars,
    scan_polars,
    to_astropy,
    to_duckdb,
    to_pandas,
    to_polars,
    write_csv,
    write_ipc,
    write_parquet,
)
from ._table.mutation import (
    append_rows,
    delete_rows,
    drop_columns,
    insert_column,
    insert_rows,
    rename_columns,
    replace_column,
    update_rows,
)
from ._table.read import (
    dataset,
    read,
    read_torch,
    reader,
    scan,
    scan_torch,
    scanner,
    schema,
)
from ._table.write import write
from ._table_engine import TABLE_BACKENDS

# Explicit Arrow synonym of ``read`` (destination-qualified symmetry).
read_arrow = read


def clear_cache() -> None:
    """Clear torchfits I/O caches (no per-table handle cache after Option A)."""
    from .cache import clear_cache as _clear_all

    _clear_all()


__all__ = [
    "TABLE_BACKENDS",
    "append_rows",
    "clear_cache",
    "FITSPolarsFrame",
    "dataset",
    "delete_rows",
    "drop_columns",
    "duckdb_query",
    "insert_column",
    "insert_rows",
    "read",
    "read_arrow",
    "read_astropy",
    "read_polars",
    "read_torch",
    "reader",
    "rename_columns",
    "replace_column",
    "scan",
    "scan_polars",
    "scan_torch",
    "scanner",
    "schema",
    "to_astropy",
    "to_duckdb",
    "to_pandas",
    "to_polars",
    "update_rows",
    "write",
    "write_csv",
    "write_ipc",
    "write_parquet",
]
