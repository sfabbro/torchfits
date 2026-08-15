"""ML-native FITS datasets and DataLoader integration.

Provides :class:`torch.utils.data.Dataset` and
:class:`~torch.utils.data.IterableDataset` implementations that wrap
``read_tensor`` / ``table.read``, plus ``fits_collate_fn`` and ``make_loader``.

Basic usage::

    from torchfits.data import FitsImageDataset, make_loader

    ds = FitsImageDataset("observations/*.fits", label_key="CLASS")
    loader = make_loader(ds, batch_size=32, num_workers=4)
    for images, labels in loader:
        ...

See [API Reference — Data Module](api.md#data-module).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .._io_engine.device import to_device
from .datasets import (
    FitsCubeDataset,
    FitsCubeIterableDataset,
    FitsImageDataset,
    FitsImageIterableDataset,
    FitsSpectrumDataset,
    FitsSpectrumIterableDataset,
    FitsStagedCutoutIterableDataset,
    FitsTensorDataset,
    FitsTensorIterableDataset,
    _buffered_shuffle,
    _resolve_rank_and_world_size,
)
from .remote import is_remote_url, prefetch_urls, resolve_local_path

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FitsTableDataset — row-indexable table catalog
# ---------------------------------------------------------------------------


class FitsTableDataset(Dataset[Any]):
    """Map-style dataset for row-indexable FITS binary tables.

    Each ``__getitem__`` returns a ``(row_dict, label)`` tuple where *label*
    is a ``torch.long`` tensor (zero when no ``labels`` are provided).
    Supports column projection and ``where=`` pushdown for filtered access.
    **Loads the full filtered table at ``__init__``** — small/medium catalogs
    only; use :class:`FitsTableIterableDataset` for large files.

    Parameters
    ----------
    path : str
        Path to the FITS file.
    hdu : int
        Table HDU index (default 1).
    columns : list[str] or None
        Column names to read.  None reads all columns.
    where : str or None
        FITS WHERE expression for row filtering (e.g. ``"MAG < 20"``).
    labels : list[int] or None
        Per-row integer labels (default 0 when omitted).
    transform : callable or None
        Optional transform applied to the row dict.
    device : str
        Torch device for tensors.
    mmap : bool or str
        Mmap policy.
    """

    def __init__(
        self,
        path: str,
        hdu: int = 1,
        columns: list[str] | None = None,
        where: str | None = None,
        labels: list[int] | None = None,
        transform: Callable[..., Any] | None = None,
        device: str = "cpu",
        mmap: bool | str = "auto",
    ) -> None:
        self.path = resolve_local_path(path)
        self.hdu = hdu
        self.columns = columns
        self.where = where
        self.transform = transform
        self.device = device
        self.mmap = mmap

        # Read all data once and index by row.
        # Small-to-medium catalogs only; see roadmap for streaming variant.
        self._data = _eager_table_columns(
            self.path,
            hdu=self.hdu,
            columns=self.columns,
            where=self.where,
            device=self.device,
            mmap=self.mmap,
        )
        self._n_rows = 0
        for v in self._data.values():
            if isinstance(v, (torch.Tensor, list)):
                self._n_rows = len(v)
                break

        if labels is not None:
            if len(labels) != self._n_rows:
                raise ValueError(
                    f"labels length {len(labels)} != n_rows {self._n_rows}"
                )
            self._labels: list[int] | None = list(labels)
        else:
            self._labels = None

    def __len__(self) -> int:
        return self._n_rows

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        row = {k: v[idx] for k, v in self._data.items()}
        if self.transform is not None:
            row = self.transform(row)
        label_val = 0 if self._labels is None else self._labels[idx]
        label = torch.tensor(label_val, dtype=torch.long)
        return row, label

    def __repr__(self) -> str:
        return (
            f"FitsTableDataset(path={self.path!r}, hdu={self.hdu}, "
            f"n_rows={self._n_rows})"
        )


def _resolve_table_mmap(mmap: bool | str) -> bool:
    return True if mmap == "auto" else bool(mmap)


def _move_table_chunk(chunk: dict[str, Any], device: str) -> dict[str, Any]:
    if str(device) == "cpu":
        return chunk
    moved: dict[str, Any] = {}
    for key, value in chunk.items():
        if isinstance(value, torch.Tensor):
            moved[key] = to_device(value, device)
        else:
            moved[key] = value
    return moved


def _eager_table_columns(
    path: str,
    *,
    hdu: int,
    columns: list[str] | None,
    where: str | None,
    device: str,
    mmap: bool | str,
) -> dict[str, Any]:
    """Load a full table into columnar storage (tensors or python lists)."""
    if where is None:
        try:
            import torchfits._C as cpp

            col_list = list(columns) if columns else []
            chunk = cpp.read_fits_table(path, hdu, col_list, _resolve_table_mmap(mmap))
            if chunk:
                return _move_table_chunk(_normalize_cpp_chunk(chunk), device)
        except Exception:
            _log.debug(
                "C++ table read failed for %r, falling back to pyarrow",
                path,
                exc_info=True,
            )

    import torchfits.table
    import pyarrow.types as pt

    pa_table = torchfits.table.read(
        path,
        hdu=hdu,
        columns=columns,
        where=where,
    )
    result: dict[str, Any] = {}
    for col_name in pa_table.column_names:
        col = pa_table.column(col_name)
        col_type = col.type
        if (
            pt.is_integer(col_type)
            or pt.is_floating(col_type)
            or pt.is_boolean(col_type)
        ):
            # Writable copy: Arrow buffers are often read-only for torch.from_numpy.
            result[col_name] = torch.as_tensor(
                col.to_numpy(zero_copy_only=False).copy()
            )
            if str(device) != "cpu":
                result[col_name] = to_device(result[col_name], device)
        else:
            result[col_name] = col.to_pylist()
    return result


def _normalize_cpp_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in chunk.items():
        if isinstance(value, torch.Tensor):
            out[name] = value
        elif isinstance(value, list):
            out[name] = value
        else:
            out[name] = value
    return out


def _tensor_columns_from_record_batch(batch: Any) -> dict[str, torch.Tensor | None]:
    """Numeric Arrow columns as tensors; None marks non-numeric columns."""
    import pyarrow.types as pt

    columns: dict[str, torch.Tensor | None] = {}
    for name in batch.schema.names:
        col = batch.column(name)
        col_type = col.type
        if (
            pt.is_integer(col_type)
            or pt.is_floating(col_type)
            or pt.is_boolean(col_type)
        ):
            columns[name] = torch.as_tensor(col.to_numpy(zero_copy_only=False).copy())
        else:
            columns[name] = None
    return columns


def _row_from_record_batch(
    batch: Any,
    row_idx: int,
    tensor_cols: dict[str, torch.Tensor | None],
    device: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for name in batch.schema.names:
        tensor_col = tensor_cols.get(name)
        if tensor_col is not None:
            value = tensor_col[row_idx]
            row[name] = to_device(value, device) if str(device) != "cpu" else value
        else:
            row[name] = batch.column(name)[row_idx].as_py()
    return row


def _row_from_torch_chunk(chunk: dict[str, Any], row_idx: int) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for name, value in chunk.items():
        if isinstance(value, torch.Tensor):
            row[name] = value[row_idx]
        elif isinstance(value, list):
            row[name] = value[row_idx]
        else:
            row[name] = value
    return row


# ---------------------------------------------------------------------------
# FitsTableIterableDataset — constant-memory table streaming
# ---------------------------------------------------------------------------


class FitsTableIterableDataset(IterableDataset[Any]):
    """Iterable dataset streaming FITS table rows via ``table.scan``.

    Yields one ``dict[str, Tensor]`` per row. NOTE: workers shard by scan
    batch index (``batch_idx % num_workers``), not by row index — fine for
    large single-file catalogs; uneven if ``batch_size`` ≫ row count.
    """

    def __init__(
        self,
        path: str,
        hdu: int = 1,
        columns: list[str] | None = None,
        where: str | None = None,
        batch_size: int = 65536,
        transform: Callable[..., Any] | None = None,
        device: str = "cpu",
        mmap: bool | str = "auto",
        shuffle_buffer_size: int | None = None,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        self.path = resolve_local_path(path)
        self.hdu = hdu
        self.columns = columns
        self.where = where
        self.batch_size = batch_size
        self.transform = transform
        self.device = device
        self.mmap = mmap
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

    def _generate(self) -> Iterator[dict[str, Any]]:
        rank, world_size = _resolve_rank_and_world_size(self.rank, self.world_size)
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        total_slots = world_size * num_workers
        slot_id = rank * num_workers + worker_id

        if self.where is None:
            import torchfits.table

            for batch_idx, chunk in enumerate(
                torchfits.table.scan_torch(
                    self.path,
                    hdu=self.hdu,
                    columns=self.columns,
                    batch_size=self.batch_size,
                    mmap=_resolve_table_mmap(self.mmap),
                    device=self.device,
                )
            ):
                if batch_idx % total_slots != slot_id:
                    continue
                if not chunk:
                    continue
                n_rows = next(
                    (v.shape[0] for v in chunk.values() if isinstance(v, torch.Tensor)),
                    0,
                )
                for row_idx in range(n_rows):
                    row = _row_from_torch_chunk(chunk, row_idx)
                    if self.transform is not None:
                        row = self.transform(row)
                    yield row
            return

        import torchfits.table

        for batch_idx, batch in enumerate(
            torchfits.table.scan(
                self.path,
                hdu=self.hdu,
                columns=self.columns,
                where=self.where,
                batch_size=self.batch_size,
                mmap=bool(self.mmap),
            )
        ):
            if batch_idx % total_slots != slot_id:
                continue
            tensor_cols = _tensor_columns_from_record_batch(batch)
            for row_idx in range(batch.num_rows):
                row = _row_from_record_batch(batch, row_idx, tensor_cols, self.device)
                if self.transform is not None:
                    row = self.transform(row)
                yield row

    def __iter__(self) -> Iterator[dict[str, Any]]:
        stream = self._generate()
        if self.shuffle_buffer_size is not None and self.shuffle_buffer_size > 1:
            stream = _buffered_shuffle(
                stream, buffer_size=self.shuffle_buffer_size, seed=self.seed
            )
        return stream

    def __repr__(self) -> str:
        return (
            f"FitsTableIterableDataset(path={self.path!r}, hdu={self.hdu}, "
            f"batch_size={self.batch_size})"
        )


# ---------------------------------------------------------------------------
# FitsCutoutDataset — patch training from cutout index table
# ---------------------------------------------------------------------------

CutoutSpec = tuple[str, int, int, int, int, int]
"""(path, hdu, x1, y1, x2, y2) half-open [x1,x2)x[y1,y2) exclusive upper bounds (matches read_subset)."""


class FitsCutoutDataset(Dataset[Any]):
    """Map-style dataset for fixed cutouts from one or more FITS images.

    Each ``__getitem__`` calls ``read_subset`` for one window. NOTE:
    same-path cutouts re-open the file each row; use ``open_subset_reader``
    when one mosaic dominates.
    """

    def __init__(
        self,
        cutouts: Sequence[CutoutSpec | tuple[str, int, int, int, int]],
        transform: Callable[..., Any] | None = None,
        device: str = "cpu",
        add_channel_dim: bool = True,
    ) -> None:
        normalized: list[CutoutSpec] = []
        for spec in cutouts:
            if len(spec) == 5:
                path, hdu, x, y, size = spec
                normalized.append((path, hdu, x, y, x + size, y + size))
            elif len(spec) == 6:
                normalized.append(tuple(spec))  # type: ignore[arg-type]
            else:
                raise ValueError(
                    "cutout must be (path, hdu, x, y, size) or "
                    "(path, hdu, x1, y1, x2, y2)"
                )
        self.cutouts = normalized
        self.transform = transform
        self.device = device
        self.add_channel_dim = add_channel_dim
        self.files = sorted({c[0] for c in self.cutouts})

    def __len__(self) -> int:
        return len(self.cutouts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        from torchfits import read_subset

        path, hdu, x1, y1, x2, y2 = self.cutouts[idx]
        # HTTP(S) uncompressed 2D uses Range cutouts; compressed/vos fall back
        # inside read_subset. Do not force full prefetch here.
        image = read_subset(path, hdu, x1, y1, x2, y2)
        if str(self.device) != "cpu":
            image = to_device(image, self.device)
        if self.add_channel_dim and image.ndim == 2:
            image = image.unsqueeze(0)
        if self.transform is not None:
            image = self.transform(image)
        return image  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"FitsCutoutDataset(n={len(self.cutouts)}, device={self.device!r})"


# ---------------------------------------------------------------------------
# fits_collate_fn — stack homogeneous tensors from batch of dicts/tensors
# ---------------------------------------------------------------------------


def fits_collate_fn(
    batch: list[Any],
) -> Any:
    """Collate a list of samples into a batch.

    - Lists of ``dict[str, Tensor]`` are stacked per key.
    - Lists of ``Tensor`` are stacked with ``torch.stack``.
    - Lists of ``(dict[str, Tensor], Tensor)`` (table row, label) pairs are
      stacked into ``(dict, labels)``.
    - Lists of ``(Tensor, Tensor)`` (image, label) pairs are stacked into
      ``(images, labels)`` tuples.
    - Ragged / VLA columns raise ``ValueError``.

    Parameters
    ----------
    batch : list
        List of samples from a torchfits dataset.

    Returns
    -------
    Collated batch (torch.Tensor, tuple, or dict).
    """
    if not batch:
        return batch

    first = batch[0]

    if isinstance(first, dict):
        out: dict[str, torch.Tensor] = {}
        for key in first:
            values = [sample[key] for sample in batch]
            if not all(isinstance(v, torch.Tensor) for v in values):
                raise ValueError(
                    f"Cannot collate non-tensor column {key!r}. "
                    f"Drop string/VLA columns or supply a custom collate_fn."
                )
            out[key] = torch.stack(values)
        return out

    if isinstance(first, tuple) and len(first) == 2:
        if isinstance(first[0], dict):
            # (dict[str, Tensor], Tensor) — table rows with labels
            stacked: dict[str, torch.Tensor] = {}
            for key in first[0]:
                values = [s[0][key] for s in batch]
                if not all(isinstance(v, torch.Tensor) for v in values):
                    raise ValueError(
                        f"Cannot collate non-tensor column {key!r}. "
                        f"Drop string/VLA columns or supply a custom collate_fn."
                    )
                stacked[key] = torch.stack(values)
            labels = torch.stack([s[1] for s in batch])
            return stacked, labels
        # (Tensor, Tensor) — image, label pairs
        images = torch.stack([s[0] for s in batch])
        labels = torch.stack([s[1] for s in batch])
        return images, labels

    if isinstance(first, torch.Tensor):
        return torch.stack(batch)

    raise TypeError(f"Unsupported sample type for collation: {type(first)}")


# ---------------------------------------------------------------------------
# make_loader — DataLoader with cache optimisation
# ---------------------------------------------------------------------------


def make_loader(
    dataset: Dataset[Any] | IterableDataset[Any],
    batch_size: int = 32,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    *,
    optimize_cache: bool = True,
    avg_file_size_mb: float = 10.0,
    **loader_kwargs: Any,
) -> DataLoader[Any]:
    """Create a DataLoader with sensible defaults and optional cache warm-up.

    When *optimize_cache* is True and the dataset exposes a ``files``
    attribute, :func:`torchfits.cache.optimize_for_dataset` is called to
    pre-warm handle and file caches.

    Parameters
    ----------
    dataset : Dataset or IterableDataset
        A torchfits dataset instance.
    batch_size : int
        Batch size.
    shuffle : bool or None
        Whether to shuffle.  Defaults to True for map-style datasets.
    num_workers : int
        Number of DataLoader worker processes.
    pin_memory : bool
        Pin memory for faster CPU→GPU transfers.
    prefetch_factor : int
        Prefetch factor for multi-worker loading.
    drop_last : bool
        Drop the last incomplete batch.
    optimize_cache : bool
        Call ``cache.optimize_for_dataset`` before creating the loader.
    avg_file_size_mb : float
        Average file size in MB used for cache sizing.
    **loader_kwargs :
        Passed through to :class:`torch.utils.data.DataLoader`.

    Returns
    -------
    DataLoader
    """
    collate_fn = loader_kwargs.pop("collate_fn", fits_collate_fn)

    if shuffle is None:
        # Map-style datasets shuffle by default; IterableDataset must not
        shuffle = not isinstance(dataset, IterableDataset)

    # prefetch_factor is only valid with num_workers > 0
    if num_workers == 0:
        prefetch_factor = None  # type: ignore[assignment]

    if optimize_cache:
        file_list = getattr(dataset, "files", None)
        if file_list:
            from torchfits.cache import optimize_for_dataset

            cache_dir = getattr(dataset, "cache_dir", None)
            # FitsCutoutDataset prefers HTTP Range cutouts — skip full prefetch.
            if not isinstance(dataset, FitsCutoutDataset):
                remote = [p for p in file_list if is_remote_url(str(p))]
                if remote:
                    prefetch_urls(remote, cache_dir=cache_dir)
                local = [
                    resolve_local_path(str(p), cache_dir=cache_dir) for p in file_list
                ]
            else:
                local = [str(p) for p in file_list if not is_remote_url(str(p))]
            if local:
                optimize_for_dataset(local, avg_file_size_mb=avg_file_size_mb)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        collate_fn=collate_fn,
        **loader_kwargs,
    )


__all__ = [
    "CutoutSpec",
    "FitsTensorDataset",
    "FitsTensorIterableDataset",
    "FitsImageDataset",
    "FitsImageIterableDataset",
    "FitsCubeDataset",
    "FitsCubeIterableDataset",
    "FitsSpectrumDataset",
    "FitsSpectrumIterableDataset",
    "FitsTableDataset",
    "FitsTableIterableDataset",
    "FitsCutoutDataset",
    "FitsStagedCutoutIterableDataset",
    "fits_collate_fn",
    "make_loader",
]
