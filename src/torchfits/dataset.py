"""Dataset utilities for loading FITS data with PyTorch DataLoader.

Provides two dataset styles:

1. FITSDataset (map-style): finite, indexable collection of FITS files (optionally
   with per-item HDU / column specs). Suitable for random access, shuffling,
   multi-worker DataLoader usage.
2. FITSIterableDataset (iterable / streaming): sequential (optionally repeatable)
   stream of FITS reads with background prefetch using a lightweight thread +
   queue. Optimised for I/O bound training loops on remote data or very large
   corpora where holding full index in memory is undesirable.

Both datasets delegate I/O to torchfits.read ensuring the same fast C++ backend
paths (including parallel column reading, pinned memory, etc.).

Design goals (R2 acceptance):
- Prefetch for iterable variant improves wall time vs no prefetch on an
  artificial latency injection test (validated in follow-up benchmark test).
- Works with standard PyTorch DataLoader best practices (pin_memory handled by
  DataLoader, not internally; avoids forking unsafe state).
- Distributed aware: shard based on (rank, world_size) if torch.distributed is
  initialised (map-style); iterable variant optionally shards epoch-wise.
"""

from __future__ import annotations

import os
import queue
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from . import read  # core reader


# ---------------------------------------------------------------------------
# Helper dataclass(es) describing dataset item specifications
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FITSItemSpec:
    """Specification for reading a single FITS item (image/cube/table).

    Use with FITSDataset or directly with read().
    """

    path: str
    hdu: int | str = 0
    columns: Sequence[str] | None = None
    start_row: int = 0
    num_rows: int | None = None
    # Generic spatial/axis cutout for images/spectra/cubes
    start: Sequence[int] | None = None
    shape: Sequence[int] | None = None
    format: str = "tensor"  # can be overridden per item
    return_metadata: bool = False

    def as_read_kwargs(self) -> dict[str, Any]:
        """Translate this spec into kwargs for torchfits.read."""
        kwargs: dict[str, Any] = {
            "hdu": self.hdu,
            "columns": list(self.columns) if self.columns else None,
            "start_row": self.start_row,
            "num_rows": self.num_rows,
            "format": self.format,
            "return_metadata": self.return_metadata,
        }
        if self.start is not None:
            kwargs["start"] = list(self.start)
        if self.shape is not None:
            kwargs["shape"] = list(self.shape)
        return kwargs


@dataclass(frozen=True)
class FITSCutoutSpec:
    """Specification for a single cutout from an image/cube HDU.

    Attributes
    ----------
    hdu : int | str
        HDU index (0-based) or name.
    start : Sequence[int]
        0-based starting coordinates.
    shape : Sequence[int]
        Shape of the cutout to read.
    device : str, default 'cpu'
        Target device for resulting tensor.
    """

    hdu: int | str
    start: Sequence[int]
    shape: Sequence[int]
    device: str = "cpu"


@dataclass(frozen=True)
class FITSMultiCutoutSpec:
    """Specification for reading multiple cutouts from potentially multiple HDUs of a single FITS/MEF file.

    path : str
        FITS file path.
    cutouts : list[FITSCutoutSpec]
        List of per-cutout specifications.
    parallel : bool, default True
        If True, issue individual cutout reads in a thread pool (I/O bound parallelism).
    max_workers : int | None
        Limit thread pool size (defaults to len(cutouts) or env TORCHFITS_CUTOUT_THREADS)
    return_dict : bool, default True
        Return mapping (hdu,start_tuple)->tensor rather than list preserving order.
    """

    path: str
    cutouts: list[FITSCutoutSpec]
    parallel: bool = True
    max_workers: int | None = None
    return_dict: bool = True

    def effective_workers(self) -> int:
        """Return the effective worker count given spec and environment."""
        if not self.parallel:
            return 1
        if self.max_workers is not None:
            return self.max_workers
        env_override = os.environ.get("TORCHFITS_CUTOUT_THREADS")
        if env_override:
            try:
                val = int(env_override)
                if val > 0:
                    return min(val, len(self.cutouts))
            except Exception:
                pass
        return min(32, len(self.cutouts))  # sane upper bound


def read_multi_cutouts(spec: FITSMultiCutoutSpec, stack: bool = False) -> Any:
    """Read multiple cutouts from a single FITS file, optionally in parallel.

    Parameters
    ----------
    spec : FITSMultiCutoutSpec
        Multi-cutout specification.
    stack : bool, default False
        If True and all outputs are tensors of same shape/dtype, stack into a single tensor.

    Returns
    -------
    Sequence[Any] | Mapping[Any, Any] | torch.Tensor
        List, dict, or stacked tensor. Stacking only occurs if all outputs are tensors of same shape/dtype.
    """
    if not spec.cutouts:
        return [] if not spec.return_dict else {}

    results_order: list[tuple[int | str, tuple[int, ...]]] = [
        (c.hdu, tuple(c.start)) for c in spec.cutouts
    ]

    def _do(c: FITSCutoutSpec):  # returns tensor for image/cube HDUs
        out = read(spec.path, hdu=c.hdu, start=list(c.start), shape=list(c.shape), device=c.device)  # type: ignore[arg-type]
        if isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[0]):
            return out[0]
        return out  # type: ignore[return-value]

    if spec.parallel and len(spec.cutouts) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        out: dict[tuple[int | str, tuple[int, ...]], torch.Tensor] = {}
        max_workers = spec.effective_workers()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {
                ex.submit(_do, c): (c.hdu, tuple(c.start)) for c in spec.cutouts
            }
            for fut in as_completed(future_map):
                key = future_map[fut]
                result = fut.result()
                if not torch.is_tensor(result):
                    raise TypeError("Cutout read did not return a tensor as expected")
                out[key] = result
        if spec.return_dict:
            # Preserve insertion order matching spec.cutouts, not completion order
            ordered = {k: out[k] for k in results_order}
            vals = list(ordered.values())
            if stack and vals and all(torch.is_tensor(v) for v in vals):
                shapes = [v.shape for v in vals if isinstance(v, torch.Tensor)]
                dtypes = [v.dtype for v in vals if isinstance(v, torch.Tensor)]
                if len(set(shapes)) == 1 and len(set(dtypes)) == 1:
                    return torch.stack([v for v in vals if torch.is_tensor(v)])
            return ordered
        else:
            seq = [out[k] for k in results_order]
            if stack and seq and all(torch.is_tensor(v) for v in seq):
                shapes = [v.shape for v in seq if isinstance(v, torch.Tensor)]
                dtypes = [v.dtype for v in seq if isinstance(v, torch.Tensor)]
                if len(set(shapes)) == 1 and len(set(dtypes)) == 1:
                    return torch.stack([v for v in seq if torch.is_tensor(v)])
            return seq
    else:
        seq = [_do(c) for c in spec.cutouts]
        if spec.return_dict:
            d = {(c.hdu, tuple(c.start)): t for c, t in zip(spec.cutouts, seq, strict=True)}
            vals = list(d.values())
            if stack and vals and all(torch.is_tensor(v) for v in vals):
                shapes = [v.shape for v in vals if isinstance(v, torch.Tensor)]
                dtypes = [v.dtype for v in vals if isinstance(v, torch.Tensor)]
                if len(set(shapes)) == 1 and len(set(dtypes)) == 1:
                    return torch.stack([v for v in vals if torch.is_tensor(v)])
            return d
        else:
            if stack and seq and all(torch.is_tensor(v) for v in seq):
                shapes = [v.shape for v in seq if isinstance(v, torch.Tensor)]
                dtypes = [v.dtype for v in seq if isinstance(v, torch.Tensor)]
                if len(set(shapes)) == 1 and len(set(dtypes)) == 1:
                    return torch.stack([v for v in seq if torch.is_tensor(v)])
            return seq


# ---------------------------------------------------------------------------
# Map-style dataset
# ---------------------------------------------------------------------------
class FITSDataset(torch.utils.data.Dataset):
    """Indexable dataset of FITS reads.

    Parameters
    ----------
    items : Sequence[Union[str, FITSItemSpec, dict]]
        Each element either a path string or specification object / dict.
    transform : callable, optional
        Applied to each sample after read().
    read_kwargs : dict, optional
        Default kwargs forwarded to torchfits.read (overridden by per-item spec).
    shard_distributed : bool, default True
        If torch.distributed is initialised, shard index range across ranks.
    """

    def __init__(
        self,
        items: Sequence[str | FITSItemSpec | dict[str, Any]],
        transform: Callable[[Any], Any] | None = None,
        read_kwargs: dict[str, Any] | None = None,
        shard_distributed: bool = True,
    ) -> None:
        if not items:
            raise ValueError("FITSDataset requires non-empty items")
        self._raw_items = list(items)
        self.transform = transform
        self.read_kwargs = read_kwargs or {}
        self.shard_distributed = shard_distributed

        # Build canonical spec list
        self._specs: list[FITSItemSpec | FITSMultiCutoutSpec] = []
        for it in self._raw_items:
            if isinstance(it, FITSItemSpec | FITSMultiCutoutSpec):
                self._specs.append(it)  # type: ignore[arg-type]
            elif isinstance(it, str):
                self._specs.append(FITSItemSpec(path=it))
            elif isinstance(it, dict):
                # Detect multi-cutout form: has 'cutouts'
                if "cutouts" in it:
                    cutouts_raw = it["cutouts"]
                    cutouts: list[FITSCutoutSpec] = []
                    for c in cutouts_raw:
                        if isinstance(c, FITSCutoutSpec):
                            cutouts.append(c)
                        elif isinstance(c, dict):
                            cutouts.append(FITSCutoutSpec(**c))
                        else:
                            raise TypeError(
                                "cutouts entries must be dict or FITSCutoutSpec"
                            )
                    self._specs.append(
                        FITSMultiCutoutSpec(
                            path=it["path"],
                            cutouts=cutouts,
                            parallel=it.get("parallel", True),
                            max_workers=it.get("max_workers"),
                            return_dict=it.get("return_dict", True),
                        )
                    )  # type: ignore[arg-type]
                else:
                    self._specs.append(
                        FITSItemSpec(
                            path=it["path"],
                            **{k: v for k, v in it.items() if k != "path"},
                        )
                    )
            else:
                raise TypeError(f"Unsupported item type: {type(it)}")

        # Distributed sharding metadata
        self._rank = 0
        self._world_size = 1
        if shard_distributed:
            try:
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    self._rank = torch.distributed.get_rank()
                    self._world_size = torch.distributed.get_world_size()
            except Exception:
                pass

        # Pre-compute indices for this rank
        if self._world_size > 1:
            self._local_indices = [
                i for i in range(len(self._specs)) if i % self._world_size == self._rank
            ]
        else:
            self._local_indices = list(range(len(self._specs)))

    def __len__(self):  # type: ignore[override]
        return len(self._local_indices)

    def _resolve_spec(self, idx: int) -> FITSItemSpec | FITSMultiCutoutSpec:
        real_idx = self._local_indices[idx]
        return self._specs[real_idx]

    def __getitem__(self, idx: int):  # type: ignore[override]
        spec = self._resolve_spec(idx)
        if isinstance(spec, FITSMultiCutoutSpec):
            data = read_multi_cutouts(spec)
        else:
            kwargs = {**self.read_kwargs, **spec.as_read_kwargs()}
            data = read(spec.path, **kwargs)
        if self.transform:
            data = self.transform(data)
        return data


# ---------------------------------------------------------------------------
# Iterable / streaming dataset with background prefetch
# ---------------------------------------------------------------------------
class FITSIterableDataset(torch.utils.data.IterableDataset):
    """Streaming iterable dataset with optional background prefetch.

    Suitable for very large file lists, remote storage, or dynamically generated
    filenames. Employs a single background thread that performs blocking I/O
    and enqueues decoded samples for consumption in the main iteration thread.

    Parameters
    ----------
    source : Iterable[Union[str, FITSItemSpec, dict]] or Callable[[], Iterable]
        Source that yields specification entries. If a callable is passed it is
        invoked at the start of each iteration (epoch) to obtain a fresh iterator.
    prefetch : int, default 2
        Max number of decoded samples to buffer ahead. Set 0 / <=1 to disable
        background thread (synchronous iteration).
    read_kwargs : dict, optional
        Default kwargs forwarded to read().
    transform : callable, optional
        Post-read transform.
    drop_last : bool, default False
        Drop final partial shard in distributed mode.
    shard_distributed : bool, default True
        If distributed initialised, split stream across ranks by position.
    timeout : float, default None
        Optional timeout (seconds) waiting for next prefetched sample.
    """

    _SENTINEL = object()

    def __init__(
        self,
        source: (
            Iterable[str | FITSItemSpec | dict[str, Any]]
            | Callable[[], Iterable[str | FITSItemSpec | dict[str, Any]]]
        ),
        prefetch: int = 2,
        read_kwargs: dict[str, Any] | None = None,
        transform: Callable[[Any], Any] | None = None,
        shard_distributed: bool = True,
        drop_last: bool = False,
        timeout: float | None = None,
    ) -> None:
        super().__init__()
        self._source = source
        self.prefetch = max(0, prefetch)
        self.read_kwargs = read_kwargs or {}
        self.transform = transform
        self.timeout = timeout
        self.drop_last = drop_last

        self._rank = 0
        self._world_size = 1
        if shard_distributed:
            try:
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    self._rank = torch.distributed.get_rank()
                    self._world_size = torch.distributed.get_world_size()
            except Exception:
                pass

    # Internal helpers -----------------------------------------------------
    def _iter_source(self) -> Iterator[FITSItemSpec]:
        src = self._source() if callable(self._source) else self._source
        iterable = cast(Iterable[str | FITSItemSpec | dict[str, Any]], src)
        for i, raw in enumerate(iterable):
            # Shard by position across distributed ranks
            if self._world_size > 1 and (i % self._world_size) != self._rank:
                continue
            if isinstance(raw, FITSItemSpec):
                yield raw
            elif isinstance(raw, str):
                yield FITSItemSpec(path=raw)
            elif isinstance(raw, dict):
                yield FITSItemSpec(
                    path=raw["path"], **{k: v for k, v in raw.items() if k != "path"}
                )
            else:
                raise TypeError(f"Unsupported source entry: {type(raw)}")

    def _producer(self, q: queue.Queue[Any], specs: Iterator[FITSItemSpec]):
        try:
            for spec in specs:
                kwargs = {**self.read_kwargs, **spec.as_read_kwargs()}
                sample = read(spec.path, **kwargs)
                if self.transform:
                    sample = self.transform(sample)
                q.put(sample)
            q.put(self._SENTINEL)
        except Exception as e:  # propagate errors to consumer
            q.put(e)
            q.put(self._SENTINEL)

    # Public API -----------------------------------------------------------
    def __iter__(self) -> Iterator[Any]:  # type: ignore[override]
        specs = self._iter_source()
        if self.prefetch <= 1:
            # synchronous path
            for spec in specs:
                kwargs = {**self.read_kwargs, **spec.as_read_kwargs()}
                sample = read(spec.path, **kwargs)
                if self.transform:
                    sample = self.transform(sample)
                yield sample
            return

        q: queue.Queue[Any] = queue.Queue(maxsize=self.prefetch)
        t = threading.Thread(target=self._producer, args=(q, specs), daemon=True)
        t.start()
        while True:
            try:
                item = q.get(timeout=self.timeout)
            except queue.Empty:
                raise TimeoutError("Timed out waiting for prefetched FITS sample") from None
            if item is self._SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    # To help DataLoader workers re-create state after fork (no persistent threads)
    def __getstate__(self):
        d = self.__dict__.copy()
        # Nothing special to strip presently
        return d

    def __setstate__(self, state):  # pragma: no cover - defensive
        self.__dict__.update(state)


__all__ = [
    "FITSItemSpec",
    "FITSDataset",
    "FITSIterableDataset",
    "FITSCutoutSpec",
    "FITSMultiCutoutSpec",
    "read_multi_cutouts",
]


# ---------------------------------------------------------------------------
# Batch reading across random list of files (mixed cutouts & table slices)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BatchReadSpec:
    """Generic batched read request.

    Accepts the union of FITSItemSpec fields but focuses on multi-file usage.

        Notes
        -----
        - columns vs col_start/col_count: If 'columns' is provided, it takes precedence.
            If 'columns' is None and 'col_start' is provided, we map contiguous column
            indices [col_start : col_start+col_count) to names using the FITS header (TTYPEi).
            If 'col_count' is None, slices to the end.
        - start/shape: For image/cube HDUs, define a cutout region using 0-based 'start' and 'shape'.
    """

    path: str
    hdu: int | str = 0
    start: Sequence[int] | None = (
        None  # spatial / spectral start (image/spectrum/cube)
    )
    shape: Sequence[int] | None = None  # spatial / spectral shape
    columns: Sequence[str] | None = None  # table columns
    start_row: int = 0  # table row window start (alias: row_start)
    num_rows: int | None = None  # table row window length (alias: row_count)
    row_start: int | None = (
        None  # user-facing alias, overrides start_row if provided
    )
    row_count: int | None = None  # user-facing alias, overrides num_rows if provided
    # Column slicing aliases (contiguous by index). If provided and 'columns' is None,
    # we will fetch header to map to column names and generate a subset list.
    col_start: int | None = None
    col_count: int | None = None  # None => to end
    format: str = "tensor"
    device: str = "cpu"  # final desired device (cpu/cuda)
    async_gpu: bool = (
        False  # request async host->gpu transfer (images/cubes only for now)
    )
    non_blocking: bool = True  # use non_blocking=True on .to()

    def as_read_kwargs(self) -> dict[str, Any]:
        """Translate this batch spec into kwargs for torchfits.read."""
        # Resolve alias overrides
        start_row = self.row_start if self.row_start is not None else self.start_row
        num_rows = self.row_count if self.row_count is not None else self.num_rows
        columns = list(self.columns) if self.columns else None
        # Apply column slicing alias only if explicit columns not provided
        if columns is None and self.col_start is not None:
            from .fits_reader import get_header  # local import to avoid cycles

            hdr = get_header(self.path, self.hdu)
            try:
                total = int(hdr.get("TFIELDS", 0))
            except Exception:
                total = 0
            if total <= 0:
                raise ValueError(
                    "Column slicing requested but no table columns found (TFIELDS<=0)"
                )
            if self.col_start < 0 or self.col_start >= total:
                raise ValueError(
                    f"col_start {self.col_start} out of range (total {total})"
                )
            # derive end index (exclusive)
            if self.col_count is None:
                end = total
            else:
                if self.col_count <= 0:
                    raise ValueError("col_count must be positive or None")
                end = min(total, self.col_start + self.col_count)
            names: list[str] = []
            for i in range(1, total + 1):
                key = f"TTYPE{i}"
                name = hdr.get(key)
                if name:
                    names.append(name)
                else:
                    names.append(f"COL{i}")  # fallback placeholder
            columns = names[self.col_start : end]
        kwargs: dict[str, Any] = {
            "hdu": self.hdu,
            "columns": columns,
            "start_row": start_row,
            "num_rows": num_rows,
            "format": self.format,
            # Always read to CPU first for async GPU pipeline; direct GPU still supported if user wants synchronous .to()
            "device": (
                "cpu"
                if (self.async_gpu and self.device.startswith("cuda"))
                else self.device
            ),
        }
        if self.start is not None:
            kwargs["start"] = list(self.start)
        if self.shape is not None:
            kwargs["shape"] = list(self.shape)
        return kwargs


def read_batch(
    specs: Sequence[BatchReadSpec],
    parallel: bool = True,
    return_dict: bool = False,
    use_stream: bool = True,
    stack: bool = False,
    preserve_headers_on_stack: bool = False,
) -> Any:
    """Read a heterogeneous batch (images/spectra/cubes/table slices) possibly from many files.

    Parameters
    ----------
    specs : sequence[BatchReadSpec]
        List of read specifications.
    parallel : bool, default True
        Use thread pool to issue reads concurrently (I/O bound).
    return_dict : bool, default False
        If True, return mapping index->data; else list preserving order.
        stack : bool, default False
                If True and all outputs are tensors of same shape/dtype, stack into a single tensor.
                - For image/cube outputs that include headers (tensor, header), stacking will operate on
                    the tensor component. By default headers are dropped from the stacked return.
    preserve_headers_on_stack : bool, default False
        When stacking and inputs were image tuples (tensor, header), return a tuple
        of (stacked_tensor, headers_list). If False, headers are dropped in the
        stacked return.

    Returns
    -------
    Sequence[Any] | Mapping[Any, Any] | torch.Tensor
        List, dict, or stacked tensor. Stacking only occurs if all outputs are tensors of same shape/dtype.
    """
    if not specs:
        return {} if return_dict else []

    # Optional dedicated CUDA stream for async transfers
    cuda_stream: torch.cuda.Stream | None = None
    if (
        use_stream
        and any(s.async_gpu and s.device.startswith("cuda") for s in specs)
        and torch.cuda.is_available()
    ):  # pragma: no cover (cuda path)
        cuda_stream = torch.cuda.Stream()

    def _maybe_async_transfer(
        data: Any, spec: BatchReadSpec
    ):  # pragma: no cover (cuda path)
        if not (
            spec.async_gpu
            and spec.device.startswith("cuda")
            and torch.cuda.is_available()
        ):
            return data
        # Handle (tensor, header) tuple or plain tensor
        header = None
        tensor = data
        if isinstance(data, tuple) and len(data) == 2 and torch.is_tensor(data[0]):
            tensor, header = data
        if not torch.is_tensor(
            tensor
        ):  # tables/dicts unsupported for async pipeline currently
            return data
        # Pin then async transfer
        if not tensor.is_pinned():
            try:
                tensor = tensor.pin_memory()
            except Exception:
                pass
        if cuda_stream is not None:
            with torch.cuda.stream(cuda_stream):
                tensor_gpu = tensor.to(spec.device, non_blocking=spec.non_blocking)
        else:
            tensor_gpu = tensor.to(spec.device, non_blocking=spec.non_blocking)
        return (tensor_gpu, header) if header is not None else tensor_gpu

    def _one(i_spec: tuple[int, BatchReadSpec]):
        i, spec = i_spec
        out = read(spec.path, **spec.as_read_kwargs())  # type: ignore[arg-type]
        # Unwrap table results to dict for stable downstream behavior
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], dict):
            out = out[0]
        # Async transfer (images / cubes). Environment override: TORCHFITS_ASYNC_GPU=1 enables by default if spec async_gpu unset.
        env_async = os.environ.get("TORCHFITS_ASYNC_GPU") == "1"
        if env_async and not spec.async_gpu:
            # heuristically promote to async if user requested globally
            spec = BatchReadSpec(**{**spec.__dict__, "async_gpu": True})  # type: ignore[arg-type]
        out = _maybe_async_transfer(out, spec)
        return i, out

    if parallel and len(specs) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = min(
            len(specs), int(os.environ.get("TORCHFITS_BATCH_THREADS", "32"))
        )
        results: dict[int, Any] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_one, (i, s)): i for i, s in enumerate(specs)}
            for fut in as_completed(futures):
                idx, data = fut.result()
                results[idx] = data
        if cuda_stream is not None:
            cuda_stream.synchronize()  # ensure all transfers done
        if return_dict:
            vals = list(results.values())
            if stack and vals:
                # Extract tensors from either raw tensors or (tensor, header) tuples
                tensors_only = []
                headers = []
                saw_header = False
                for v in vals:
                    if isinstance(v, tuple) and len(v) == 2 and torch.is_tensor(v[0]):
                        tensors_only.append(v[0])
                        headers.append(v[1])
                        saw_header = True
                    elif torch.is_tensor(v):
                        tensors_only.append(v)
                        headers.append(None)
                    else:
                        tensors_only = []
                        break
                if tensors_only and len(tensors_only) == len(vals):
                    shapes = [t.shape for t in tensors_only]
                    dtypes = [t.dtype for t in tensors_only]
                    if len(set(shapes)) == 1 and len(set(dtypes)) == 1:
                        stacked = torch.stack(tensors_only)
                        if preserve_headers_on_stack and saw_header:
                            return (stacked, headers)
                        return stacked
            return results
        seq = [results[i] for i in range(len(specs))]
        if stack and seq:
            tensors_only = []
            headers = []
            saw_header = False
            for v in seq:
                if isinstance(v, tuple) and len(v) == 2 and torch.is_tensor(v[0]):
                    tensors_only.append(v[0])
                    headers.append(v[1])
                    saw_header = True
                elif torch.is_tensor(v):
                    tensors_only.append(v)
                    headers.append(None)
                else:
                    tensors_only = []
                    break
            if tensors_only and len(tensors_only) == len(seq):
                shapes = [t.shape for t in tensors_only]
                dtypes = [t.dtype for t in tensors_only]
                if len(set(shapes)) == 1 and len(set(dtypes)) == 1:
                    stacked = torch.stack(tensors_only)
                    if preserve_headers_on_stack and saw_header:
                        return (stacked, headers)
                    return stacked
        return seq
    else:
        seq = []
        for i, s in enumerate(specs):
            _, data = _one((i, s))
            seq.append(data)
        if cuda_stream is not None:
            cuda_stream.synchronize()
        if return_dict:
            d = {i: v for i, v in enumerate(seq)}
            vals = list(d.values())
            if stack and vals:
                tensors_only = []
                headers = []
                saw_header = False
                for v in vals:
                    if isinstance(v, tuple) and len(v) == 2 and torch.is_tensor(v[0]):
                        tensors_only.append(v[0])
                        headers.append(v[1])
                        saw_header = True
                    elif torch.is_tensor(v):
                        tensors_only.append(v)
                        headers.append(None)
                    else:
                        tensors_only = []
                        break
                if tensors_only and len(tensors_only) == len(vals):
                    shapes = [t.shape for t in tensors_only]
                    dtypes = [t.dtype for t in tensors_only]
                    if len(set(shapes)) == 1 and len(set(dtypes)) == 1:
                        stacked = torch.stack(tensors_only)
                        if preserve_headers_on_stack and saw_header:
                            return (stacked, headers)
                        return stacked
            return d
        if stack and seq:
            tensors_only = []
            headers = []
            saw_header = False
            for v in seq:
                if isinstance(v, tuple) and len(v) == 2 and torch.is_tensor(v[0]):
                    tensors_only.append(v[0])
                    headers.append(v[1])
                    saw_header = True
                elif torch.is_tensor(v):
                    tensors_only.append(v)
                    headers.append(None)
                else:
                    tensors_only = []
                    break
            if tensors_only and len(tensors_only) == len(seq):
                shapes = [t.shape for t in tensors_only]
                dtypes = [t.dtype for t in tensors_only]
                if len(set(shapes)) == 1 and len(set(dtypes)) == 1:
                    stacked = torch.stack(tensors_only)
                    if preserve_headers_on_stack and saw_header:
                        return (stacked, headers)
                    return stacked
        return seq


def generate_random_cutout_specs(
    paths: Sequence[str],
    hdu: int = 0,
    shape: Sequence[int] = (32, 32),
    n: int = 8,
    device: str = "cpu",
) -> list[BatchReadSpec]:
    """Generate random spatial cutout specs across multiple files (spectra/images/cubes).

    NOTE: For cubes (>2D) only the leading dimensions covered by provided shape are sampled; the remaining
    dimensions (if shape shorter) are fully read.
    """
    import random

    from . import get_dims

    specs: list[BatchReadSpec] = []
    for _ in range(n):
        p = random.choice(paths)
        dims = get_dims(p, hdu)
        # Ensure shape broadcast
        if len(shape) > len(dims):
            raise ValueError("Provided shape has more dimensions than target HDU")
        start = []
        for i, sh in enumerate(shape):
            max_start = max(0, dims[i] - sh)
            start.append(0 if max_start == 0 else random.randint(0, max_start))
        specs.append(
            BatchReadSpec(path=p, hdu=hdu, start=start, shape=shape, device=device)
        )
    return specs


__all__ += [
    "BatchReadSpec",
    "read_batch",
    "generate_random_cutout_specs",
]


# ---------------------------------------------------------------------------
# Table cutout semantics (row/column subsets) convenience layer
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TableCutoutSpec:
    """Specification describing a row/column subset of a table HDU."""

    path: str
    hdu: int | str = 1  # tables typically at HDU 1
    row_start: int = 0
    row_count: int | None = None
    columns: Sequence[str] | None = None
    # Column slicing aliases (contiguous by index). Applied only if 'columns' not provided.
    col_start: int | None = None
    col_count: int | None = None  # None => to end
    format: str = (
        "tensor"  # 'tensor' yields dict of column->tensor; 'table' yields FitsTable
    )
    device: str = "cpu"
    return_metadata: bool = False
    # When True and format=='tensor', also return per-column boolean null masks derived from TNULLn header keys.
    return_null_masks: bool = False

    def as_read_kwargs(self) -> dict[str, Any]:
        """Translate this spec into kwargs for torchfits.read."""
        columns = list(self.columns) if self.columns else None
        if columns is None and self.col_start is not None:
            from .fits_reader import get_header

            hdr = get_header(self.path, self.hdu)
            try:
                total = int(hdr.get("TFIELDS", 0))
            except Exception:
                total = 0
            if total <= 0:
                raise ValueError(
                    "Column slicing requested but no table columns found (TFIELDS<=0)"
                )
            if self.col_start < 0 or self.col_start >= total:
                raise ValueError(
                    f"col_start {self.col_start} out of range (total {total})"
                )
            if self.col_count is None:
                end = total
            else:
                if self.col_count <= 0:
                    raise ValueError("col_count must be positive or None")
                end = min(total, self.col_start + self.col_count)
            names: list[str] = []
            for i in range(1, total + 1):
                key = f"TTYPE{i}"
                name = hdr.get(key)
                if name:
                    names.append(name)
                else:
                    names.append(f"COL{i}")
            columns = names[self.col_start : end]
        return {
            "hdu": self.hdu,
            "columns": columns,
            "start_row": self.row_start,
            "num_rows": self.row_count,
            "format": self.format,
            "device": self.device,
            "return_metadata": self.return_metadata,
        }


def read_table_cutout(
    path: str,
    hdu: int | str = 1,
    row_start: int = 0,
    row_count: int | None = None,
    columns: Sequence[str] | None = None,
    *,
    col_start: int | None = None,
    col_count: int | None = None,
    format: str = "tensor",
    device: str = "cpu",
    return_metadata: bool = False,
    return_null_masks: bool = False,
):
    """Read a subset of table rows (and optionally columns) as a lightweight 'cutout'.

    Parameters
    ----------
    path : str
        FITS file path.
    hdu : int | str, default 1
        Table HDU index (0-based) or name.
    row_start : int, default 0
        Starting row index (0-based).
    row_count : int, optional
        Number of rows to read (None means to end).
    columns : sequence[str], optional
        Column subset to select.
    format : str, default 'tensor'
        'tensor' (dict of tensors) or 'table' for FitsTable.
    device : str, default 'cpu'
        Target device for tensors.
    return_metadata : bool, default False
        Include column metadata (for 'table' format or advanced usage).
    return_null_masks : bool, default False
        When True and format=='tensor', also return a dict of boolean masks per column where True marks nulls
        according to TNULLn header values. Returns a tuple (data_dict, masks_dict).
    """
    if columns is not None and col_start is not None:
        raise ValueError("Provide either columns or col_start/col_count, not both")
    resolved_cols = list(columns) if columns else None
    if resolved_cols is None and col_start is not None:
        from .fits_reader import get_header

        hdr = get_header(path, hdu)
        try:
            total = int(hdr.get("TFIELDS", 0))
        except Exception:
            total = 0
        if total <= 0:
            raise ValueError(
                "Column slicing requested but table has no columns (TFIELDS<=0)"
            )
        if col_start < 0 or col_start >= total:
            raise ValueError(f"col_start {col_start} out of range (total {total})")
        if col_count is None:
            end = total
        else:
            if col_count <= 0:
                raise ValueError("col_count must be positive or None")
            end = min(total, col_start + col_count)
        names: list[str] = []
        for i in range(1, total + 1):
            key = f"TTYPE{i}"
            name = hdr.get(key)
            names.append(name if name else f"COL{i}")
        resolved_cols = names[col_start:end]
    kwargs = {
        "hdu": hdu,
        "start_row": row_start,
        "num_rows": row_count,
        "columns": resolved_cols,
        "format": format,
        "device": device,
        "return_metadata": return_metadata,
    }
    res = read(path, **kwargs)  # type: ignore[arg-type]
    # For tensor format, unwrap (dict, header) to dict to match convenience API expectations
    if (
        format == "tensor"
        and isinstance(res, tuple)
        and len(res) == 2
        and isinstance(res[0], dict)
    ):
        data_dict = res[0]
        if return_null_masks:
            from .fits_reader import _build_null_masks as _build_masks
            from .fits_reader import get_header as _get_header

            hdr = _get_header(path, hdu)
            masks = _build_masks(data_dict, hdr)
            return data_dict, masks
        return data_dict
    return res


def read_multi_table_cutouts(
    specs: Sequence[TableCutoutSpec],
    parallel: bool = True,
    return_dict: bool = False,
    stack: bool = False,
) -> Any:
    """Read multiple table cutouts (row/column subsets) potentially in parallel.

    Falls back to sequential if only one spec or parallel=False.
    If stack=True and all outputs are dicts with same keys and per-key tensors have same shape/dtype, stacks per key.

    Returns
    -------
    Sequence[Any] | Mapping[Any, Any] | dict[str, torch.Tensor]
        List, dict, or dict of stacked tensors. Stacking only occurs if all outputs are dicts with same keys and per-key tensors have same shape/dtype.
    """
    if not specs:
        return {} if return_dict else []

    def _one(idx_spec: tuple[int, TableCutoutSpec]):
        i, s = idx_spec
        out = read(s.path, **s.as_read_kwargs())  # type: ignore[arg-type]
        # Unwrap table (dict, header) to dict
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], dict):
            data_only = out[0]
            if getattr(s, "return_null_masks", False):
                from .fits_reader import _build_null_masks as _build_masks
                from .fits_reader import get_header as _get_header

                hdr = _get_header(s.path, s.hdu)
                masks = _build_masks(data_only, hdr)
                return i, (data_only, masks)
            return i, data_only
        return i, out

    if parallel and len(specs) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = min(
            len(specs), int(os.environ.get("TORCHFITS_TABLE_CUTOUT_THREADS", "32"))
        )
        out: dict[int, Any] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {ex.submit(_one, (i, s)): i for i, s in enumerate(specs)}
            for fut in as_completed(fut_map):
                idx, data = fut.result()
                out[idx] = data
        if return_dict:
            vals = list(out.values())
            # Stack when values are dicts
            if stack and vals and all(isinstance(v, dict) for v in vals):
                keys = set(vals[0].keys())
                if all(set(v.keys()) == keys for v in vals):
                    stacked = {}
                    for k in keys:
                        arrs = [v[k] for v in vals]
                        if arrs and all(torch.is_tensor(a) for a in arrs):
                            shapes = [
                                a.shape for a in arrs if isinstance(a, torch.Tensor)
                            ]
                            dtypes = [
                                a.dtype for a in arrs if isinstance(a, torch.Tensor)
                            ]
                            if len(set(shapes)) == 1 and len(set(dtypes)) == 1:
                                stacked[k] = torch.stack(
                                    [a for a in arrs if torch.is_tensor(a)]
                                )
                            else:
                                break
                    if len(stacked) == len(keys):
                        return stacked
            # Stack when values are (data_dict, masks_dict) tuples
            if (
                stack
                and vals
                and all(
                    isinstance(v, tuple)
                    and len(v) == 2
                    and all(isinstance(x, dict) for x in v)
                    for v in vals
                )
            ):
                data_vals = [v[0] for v in vals]
                mask_vals = [v[1] for v in vals]
                keys = set(data_vals[0].keys())
                if all(set(v.keys()) == keys for v in data_vals) and all(
                    set(v.keys()) == keys for v in mask_vals
                ):
                    stacked_data, stacked_masks = {}, {}
                    for k in keys:
                        darrs = [v[k] for v in data_vals]
                        marrs = [v[k] for v in mask_vals]
                        if (
                            darrs
                            and all(torch.is_tensor(a) for a in darrs)
                            and marrs
                            and all(
                                torch.is_tensor(a) and a.dtype == torch.bool
                                for a in marrs
                            )
                        ):
                            dshapes = [a.shape for a in darrs]
                            ddtypes = [a.dtype for a in darrs]
                            mshapes = [a.shape for a in marrs]
                            if (
                                len(set(dshapes)) == 1
                                and len(set(ddtypes)) == 1
                                and len(set(mshapes)) == 1
                            ):
                                stacked_data[k] = torch.stack(darrs)
                                stacked_masks[k] = torch.stack(marrs)
                            else:
                                break
                    if len(stacked_data) == len(keys) and len(stacked_masks) == len(
                        keys
                    ):
                        return stacked_data, stacked_masks
            return out
        seq = [out[i] for i in range(len(specs))]
        if stack and seq and all(isinstance(v, dict) for v in seq):
            keys = set(seq[0].keys())
            if all(set(v.keys()) == keys for v in seq):
                stacked = {}
                for k in keys:
                    arrs = [v[k] for v in seq]
                    if arrs and all(torch.is_tensor(a) for a in arrs):
                        shapes = [a.shape for a in arrs if isinstance(a, torch.Tensor)]
                        dtypes = [a.dtype for a in arrs if isinstance(a, torch.Tensor)]
                        if len(set(shapes)) == 1 and len(set(dtypes)) == 1:
                            stacked[k] = torch.stack(
                                [a for a in arrs if torch.is_tensor(a)]
                            )
                        else:
                            break
                if len(stacked) == len(keys):
                    return stacked
        # Stack when entries are (data_dict, masks_dict)
        if (
            stack
            and seq
            and all(
                isinstance(v, tuple)
                and len(v) == 2
                and all(isinstance(x, dict) for x in v)
                for v in seq
            )
        ):
            data_seq = [v[0] for v in seq]
            mask_seq = [v[1] for v in seq]
            keys = set(data_seq[0].keys())
            if all(set(v.keys()) == keys for v in data_seq) and all(
                set(v.keys()) == keys for v in mask_seq
            ):
                stacked_data, stacked_masks = {}, {}
                for k in keys:
                    darrs = [v[k] for v in data_seq]
                    marrs = [v[k] for v in mask_seq]
                    if (
                        darrs
                        and all(torch.is_tensor(a) for a in darrs)
                        and marrs
                        and all(
                            torch.is_tensor(a) and a.dtype == torch.bool for a in marrs
                        )
                    ):
                        dshapes = [a.shape for a in darrs]
                        ddtypes = [a.dtype for a in darrs]
                        mshapes = [a.shape for a in marrs]
                        if (
                            len(set(dshapes)) == 1
                            and len(set(ddtypes)) == 1
                            and len(set(mshapes)) == 1
                        ):
                            stacked_data[k] = torch.stack(darrs)
                            stacked_masks[k] = torch.stack(marrs)
                        else:
                            break
                if len(stacked_data) == len(keys) and len(stacked_masks) == len(keys):
                    return stacked_data, stacked_masks
        return seq
    else:
        seq = []
        for i, s in enumerate(specs):
            _, data = _one((i, s))
            seq.append(data)
        if return_dict:
            d = {i: v for i, v in enumerate(seq)}
            vals = list(d.values())
            if stack and vals and all(isinstance(v, dict) for v in vals):
                keys = set(vals[0].keys())
                if all(set(v.keys()) == keys for v in vals):
                    stacked = {}
                    for k in keys:
                        arrs = [v[k] for v in vals]
                        if arrs and all(torch.is_tensor(a) for a in arrs):
                            shapes = [
                                a.shape for a in arrs if isinstance(a, torch.Tensor)
                            ]
                            dtypes = [
                                a.dtype for a in arrs if isinstance(a, torch.Tensor)
                            ]
                            if len(set(shapes)) == 1 and len(set(dtypes)) == 1:
                                stacked[k] = torch.stack(
                                    [a for a in arrs if torch.is_tensor(a)]
                                )
                            else:
                                break
                    if len(stacked) == len(keys):
                        return stacked
            # Stack when values are (data_dict, masks_dict)
            if (
                stack
                and vals
                and all(
                    isinstance(v, tuple)
                    and len(v) == 2
                    and all(isinstance(x, dict) for x in v)
                    for v in vals
                )
            ):
                data_vals = [v[0] for v in vals]
                mask_vals = [v[1] for v in vals]
                keys = set(data_vals[0].keys())
                if all(set(v.keys()) == keys for v in data_vals) and all(
                    set(v.keys()) == keys for v in mask_vals
                ):
                    stacked_data, stacked_masks = {}, {}
                    for k in keys:
                        darrs = [v[k] for v in data_vals]
                        marrs = [v[k] for v in mask_vals]
                        if (
                            darrs
                            and all(torch.is_tensor(a) for a in darrs)
                            and marrs
                            and all(
                                torch.is_tensor(a) and a.dtype == torch.bool
                                for a in marrs
                            )
                        ):
                            dshapes = [a.shape for a in darrs]
                            ddtypes = [a.dtype for a in darrs]
                            mshapes = [a.shape for a in marrs]
                            if (
                                len(set(dshapes)) == 1
                                and len(set(ddtypes)) == 1
                                and len(set(mshapes)) == 1
                            ):
                                stacked_data[k] = torch.stack(darrs)
                                stacked_masks[k] = torch.stack(marrs)
                            else:
                                break
                    if len(stacked_data) == len(keys) and len(stacked_masks) == len(
                        keys
                    ):
                        return stacked_data, stacked_masks
            return d
        if stack and seq and all(isinstance(v, dict) for v in seq):
            keys = set(seq[0].keys())
            if all(set(v.keys()) == keys for v in seq):
                stacked = {}
                for k in keys:
                    arrs = [v[k] for v in seq]
                    if arrs and all(torch.is_tensor(a) for a in arrs):
                        shapes = [a.shape for a in arrs if isinstance(a, torch.Tensor)]
                        dtypes = [a.dtype for a in arrs if isinstance(a, torch.Tensor)]
                        if len(set(shapes)) == 1 and len(set(dtypes)) == 1:
                            stacked[k] = torch.stack(
                                [a for a in arrs if torch.is_tensor(a)]
                            )
                        else:
                            break
                if len(stacked) == len(keys):
                    return stacked
        # Stack when entries are (data_dict, masks_dict)
        if (
            stack
            and seq
            and all(
                isinstance(v, tuple)
                and len(v) == 2
                and all(isinstance(x, dict) for x in v)
                for v in seq
            )
        ):
            data_seq = [v[0] for v in seq]
            mask_seq = [v[1] for v in seq]
            keys = set(data_seq[0].keys())
            if all(set(v.keys()) == keys for v in data_seq) and all(
                set(v.keys()) == keys for v in mask_seq
            ):
                stacked_data, stacked_masks = {}, {}
                for k in keys:
                    darrs = [v[k] for v in data_seq]
                    marrs = [v[k] for v in mask_seq]
                    if (
                        darrs
                        and all(torch.is_tensor(a) for a in darrs)
                        and marrs
                        and all(
                            torch.is_tensor(a) and a.dtype == torch.bool for a in marrs
                        )
                    ):
                        dshapes = [a.shape for a in darrs]
                        ddtypes = [a.dtype for a in darrs]
                        mshapes = [a.shape for a in marrs]
                        if (
                            len(set(dshapes)) == 1
                            and len(set(ddtypes)) == 1
                            and len(set(mshapes)) == 1
                        ):
                            stacked_data[k] = torch.stack(darrs)
                            stacked_masks[k] = torch.stack(marrs)
                        else:
                            break
                if len(stacked_data) == len(keys) and len(stacked_masks) == len(keys):
                    return stacked_data, stacked_masks
        return seq


__all__ += [
    "TableCutoutSpec",
    "read_table_cutout",
    "read_multi_table_cutouts",
]
