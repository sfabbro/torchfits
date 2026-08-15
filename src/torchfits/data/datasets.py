"""Map/iterable Datasets for IMAGE tensors, images, cubes, and spectra."""

from __future__ import annotations

import glob as _glob
import os
import random
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch
from torch.utils.data import Dataset, IterableDataset

from .remote import is_remote_url, prefetch_urls, resolve_local_path

HduRef = int | str
HduSpec = HduRef | Sequence[HduRef]


def _resolve_rank_and_world_size(
    rank: int | None = None, world_size: int | None = None
) -> tuple[int, int]:
    """Resolve distributed rank and world size across explicit inputs, env vars, and torch.distributed."""
    if rank is not None and world_size is not None:
        return max(0, int(rank)), max(1, int(world_size))

    r = rank
    w = world_size

    # 1. Standard distributed cluster env vars (torchrun / DeepSpeed / FSDP / SLURM)
    if r is None:
        for env_var in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
            val = os.environ.get(env_var)
            if val is not None and val.isdigit():
                r = int(val)
                break
    if w is None:
        for env_var in ("WORLD_SIZE", "SLURM_NTASKS", "SLURM_NPROCS"):
            val = os.environ.get(env_var)
            if val is not None and val.isdigit():
                w = int(val)
                break

    # 2. torch.distributed if initialized
    if (
        (r is None or w is None)
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        if r is None:
            r = torch.distributed.get_rank()
        if w is None:
            w = torch.distributed.get_world_size()

    return (0 if r is None else max(0, int(r))), (1 if w is None else max(1, int(w)))


def _shard_sequence(seq: list[Any], rank: int, world_size: int) -> list[Any]:
    """Partition a sequence across distributed ranks."""
    if world_size <= 1 or not seq:
        return seq
    return seq[rank::world_size]


def _buffered_shuffle(
    iterator: Iterator[Any], buffer_size: int = 1000, seed: int = 0
) -> Iterator[Any]:
    """Reservoir streaming shuffle buffer with O(buffer_size) memory."""
    if buffer_size <= 1:
        yield from iterator
        return

    rng = random.Random(seed)
    buffer: list[Any] = []

    for item in iterator:
        if len(buffer) < buffer_size:
            buffer.append(item)
        else:
            idx = rng.randint(0, len(buffer) - 1)
            yield buffer[idx]
            buffer[idx] = item

    rng.shuffle(buffer)
    yield from buffer


def _resolve_paths(paths: str | list[str]) -> list[str]:
    if isinstance(paths, str):
        if is_remote_url(paths):
            return [paths]
        paths = sorted(_glob.glob(paths)) or [paths]
    return list(paths)


def _as_hdu_list(hdu: HduSpec) -> list[HduRef]:
    if isinstance(hdu, (int, str)):
        return [hdu]
    out = list(hdu)
    if not out:
        raise ValueError("hdu sequence must be non-empty")
    return out


def _arm_name(hdu: HduRef) -> str:
    return str(hdu)


def _local_read_path(
    path: str,
    *,
    prefetch_ahead: Sequence[str] | None = None,
    cache_dir: Path | None = None,
) -> str:
    if prefetch_ahead:
        prefetch_urls(prefetch_ahead, cache_dir=cache_dir)
    return resolve_local_path(path, cache_dir=cache_dir)


def _read_image(
    path: str,
    hdu: HduRef,
    *,
    device: str,
    mmap: bool | str,
) -> torch.Tensor:
    from torchfits import read

    tensor = read(path, hdu=hdu, mode="image", device=device, mmap=mmap)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected IMAGE tensor from hdu={hdu!r}, got {type(tensor)}")
    return tensor


def _stack_flux(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) != 1:
        raise ValueError(
            f"multi-HDU flux channels require identical shapes; got {sorted(shapes)}"
        )
    return torch.stack(tensors, dim=0)


def _read_flux_stack(
    path: str,
    hdus: list[HduRef],
    *,
    device: str,
    mmap: bool | str,
) -> torch.Tensor:
    return _stack_flux([_read_image(path, h, device=device, mmap=mmap) for h in hdus])


def _optional_companion(
    path: str,
    hdus: list[HduRef] | None,
    *,
    device: str,
    mmap: bool | str,
) -> torch.Tensor | None:
    if hdus is None:
        return None
    return _read_flux_stack(path, hdus, device=device, mmap=mmap)


def _pack_payload(
    flux: torch.Tensor,
    ivar: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if ivar is None and mask is None:
        return flux
    out: dict[str, torch.Tensor] = {"flux": flux}
    if ivar is not None:
        out["ivar"] = ivar
    if mask is not None:
        out["mask"] = mask
    return out


def _load_image_payload(
    path: str,
    *,
    hdus: list[Any],
    ivar_hdus: list[Any] | None,
    mask_hdus: list[Any] | None,
    device: str,
    mmap: bool | str,
    add_channel_dim: bool,
    transform: Callable[..., Any] | None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    flux = _read_flux_stack(path, hdus, device=device, mmap=mmap)
    if add_channel_dim and flux.ndim == 2:
        flux = flux.unsqueeze(0)
    ivar = _optional_companion(path, ivar_hdus, device=device, mmap=mmap)
    mask = _optional_companion(path, mask_hdus, device=device, mmap=mmap)
    payload = _pack_payload(flux, ivar, mask)
    if transform is not None:
        payload = transform(payload)
    return payload


class FitsTensorDataset(Dataset[Any]):
    """General N-D IMAGE HDU → tensor (any rank).

    Multi-HDU ``hdu=[…]`` stacks **flux** channels on dim 0. Optional
    ``ivar_hdu`` / ``mask_hdu`` are companion tensors (never flux channels).
    """

    def __init__(
        self,
        paths: str | list[str],
        hdu: HduSpec = 0,
        ivar_hdu: HduSpec | None = None,
        mask_hdu: HduSpec | None = None,
        label_key: str | None = None,
        labels: list[int] | None = None,
        transform: Callable[..., Any] | None = None,
        device: str = "cpu",
        mmap: bool | str = True,
        add_channel_dim: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.files = _resolve_paths(paths)
        self.hdus = _as_hdu_list(hdu)
        self.ivar_hdus = None if ivar_hdu is None else _as_hdu_list(ivar_hdu)
        self.mask_hdus = None if mask_hdu is None else _as_hdu_list(mask_hdu)
        if self.ivar_hdus is not None and len(self.ivar_hdus) != len(self.hdus):
            raise ValueError("ivar_hdu must match hdu arity")
        if self.mask_hdus is not None and len(self.mask_hdus) != len(self.hdus):
            raise ValueError("mask_hdu must match hdu arity")
        self.transform = transform
        self.device = device
        self.mmap = mmap
        self.add_channel_dim = add_channel_dim
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.hdu = self.hdus[0] if len(self.hdus) == 1 else self.hdus

        if labels is not None:
            if len(labels) != len(self.files):
                raise ValueError(
                    f"labels length {len(labels)} != files length {len(self.files)}"
                )
            self._labels = list(labels)
        elif label_key is not None:
            from torchfits import read_keys

            self._labels = [
                int(
                    read_keys(
                        _local_read_path(f, cache_dir=self.cache_dir),
                        [label_key],
                        hdu=self.hdus[0],
                    )[label_key]
                )
                for f in self.files
            ]
        else:
            self._labels = [0] * len(self.files)

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, path: str) -> torch.Tensor | dict[str, torch.Tensor]:
        return _load_image_payload(
            path,
            hdus=self.hdus,
            ivar_hdus=self.ivar_hdus,
            mask_hdus=self.mask_hdus,
            device=self.device,
            mmap=self.mmap,
            add_channel_dim=self.add_channel_dim,
            transform=self.transform,
        )

    def __getitem__(self, idx: int) -> tuple[Any, torch.Tensor]:
        ahead = self.files[idx + 1 : idx + 3]
        path = _local_read_path(
            self.files[idx], prefetch_ahead=ahead, cache_dir=self.cache_dir
        )
        payload = self._load(path)
        label = torch.tensor(self._labels[idx], dtype=torch.long)
        return payload, label

    def __repr__(self) -> str:
        return (
            f"FitsTensorDataset(n={len(self.files)}, hdu={self.hdu!r}, "
            f"device={self.device!r})"
        )


class FitsImageDataset(FitsTensorDataset):
    """2D image peer: multi-band HDUs → ``[C,H,W]``; ``add_channel_dim`` default True."""

    def __init__(
        self,
        paths: str | list[str],
        hdu: HduSpec = 0,
        *,
        add_channel_dim: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(paths, hdu=hdu, add_channel_dim=add_channel_dim, **kwargs)

    def __repr__(self) -> str:
        return (
            f"FitsImageDataset(n={len(self.files)}, hdu={self.hdu!r}, "
            f"device={self.device!r})"
        )


class FitsCubeDataset(FitsTensorDataset):
    """3D+ cube peer (optional leading-axis ``slice_index``)."""

    def __init__(
        self,
        paths: str | list[str],
        hdu: HduSpec = 0,
        slice_index: int | None = None,
        *,
        add_channel_dim: bool = False,
        **kwargs: Any,
    ) -> None:
        self.slice_index = slice_index
        super().__init__(paths, hdu=hdu, add_channel_dim=add_channel_dim, **kwargs)

    def __getitem__(self, idx: int) -> tuple[Any, torch.Tensor]:
        payload, label = super().__getitem__(idx)
        if self.slice_index is None:
            return payload, label
        if isinstance(payload, dict):
            sliced = {
                key: value[self.slice_index]
                if isinstance(value, torch.Tensor)
                else value
                for key, value in payload.items()
            }
            return sliced, label
        return payload[self.slice_index], label

    def __repr__(self) -> str:
        return (
            f"FitsCubeDataset(n={len(self.files)}, hdu={self.hdu!r}, "
            f"slice_index={self.slice_index!r})"
        )


class FitsTensorIterableDataset(IterableDataset[Any]):
    """Sharded iterable general N-D IMAGE reader."""

    def __init__(
        self,
        paths: str | list[str],
        hdu: HduSpec = 0,
        ivar_hdu: HduSpec | None = None,
        mask_hdu: HduSpec | None = None,
        transform: Callable[..., Any] | None = None,
        device: str = "cpu",
        mmap: bool | str = True,
        shuffle: bool = False,
        shuffle_buffer_size: int | None = None,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
        add_channel_dim: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.files = _resolve_paths(paths)
        self.hdus = _as_hdu_list(hdu)
        self.ivar_hdus = None if ivar_hdu is None else _as_hdu_list(ivar_hdu)
        self.mask_hdus = None if mask_hdu is None else _as_hdu_list(mask_hdu)
        self.transform = transform
        self.device = device
        self.mmap = mmap
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.add_channel_dim = add_channel_dim
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.hdu = self.hdus[0] if len(self.hdus) == 1 else self.hdus

    def _generate(self) -> Iterator[Any]:
        rank, world_size = _resolve_rank_and_world_size(self.rank, self.world_size)
        sharded_files = _shard_sequence(self.files, rank, world_size)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            indices = list(range(len(sharded_files)))
            worker_seed = self.seed
        else:
            total = len(sharded_files)
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = total // num_workers
            remainder = total % num_workers
            start = worker_id * per_worker + min(worker_id, remainder)
            size = per_worker + (1 if worker_id < remainder else 0)
            indices = list(range(start, start + size))
            worker_seed = self.seed + worker_id

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(worker_seed)
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]

        for i, idx in enumerate(indices):
            ahead = [
                sharded_files[indices[j]]
                for j in range(i + 1, min(i + 3, len(indices)))
            ]
            path = _local_read_path(
                sharded_files[idx], prefetch_ahead=ahead, cache_dir=self.cache_dir
            )
            yield _load_image_payload(
                path,
                hdus=self.hdus,
                ivar_hdus=self.ivar_hdus,
                mask_hdus=self.mask_hdus,
                device=self.device,
                mmap=self.mmap,
                add_channel_dim=self.add_channel_dim,
                transform=self.transform,
            )

    def __iter__(self) -> Iterator[Any]:
        stream = self._generate()
        if self.shuffle_buffer_size is not None and self.shuffle_buffer_size > 1:
            stream = _buffered_shuffle(
                stream, buffer_size=self.shuffle_buffer_size, seed=self.seed
            )
        return stream

    def __repr__(self) -> str:
        return (
            f"FitsTensorIterableDataset(n={len(self.files)}, hdu={self.hdu!r}, "
            f"device={self.device!r})"
        )


class FitsImageIterableDataset(FitsTensorIterableDataset):
    """2D image iterable peer (``add_channel_dim`` default True)."""

    def __init__(
        self,
        paths: str | list[str],
        hdu: HduSpec = 0,
        *,
        add_channel_dim: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(paths, hdu=hdu, add_channel_dim=add_channel_dim, **kwargs)

    def __repr__(self) -> str:
        return (
            f"FitsImageIterableDataset(n={len(self.files)}, hdu={self.hdu!r}, "
            f"device={self.device!r})"
        )


class FitsCubeIterableDataset(FitsTensorIterableDataset):
    """3D+ datacube streaming peer with optional leading-axis ``slice_index``."""

    def __init__(
        self,
        paths: str | list[str],
        hdu: HduSpec = 0,
        slice_index: int | None = None,
        *,
        add_channel_dim: bool = False,
        **kwargs: Any,
    ) -> None:
        self.slice_index = slice_index
        super().__init__(paths, hdu=hdu, add_channel_dim=add_channel_dim, **kwargs)

    def __iter__(self) -> Iterator[Any]:
        for payload in super().__iter__():
            if self.slice_index is None:
                yield payload
            elif isinstance(payload, dict):
                yield {
                    key: value[self.slice_index]
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in payload.items()
                }
            else:
                yield payload[self.slice_index]

    def __repr__(self) -> str:
        return (
            f"FitsCubeIterableDataset(n={len(self.files)}, hdu={self.hdu!r}, "
            f"slice_index={self.slice_index!r})"
        )


class FitsSpectrumDataset(Dataset[Any]):
    """1D spectra (IMAGE or table column), optional multi-arm layouts.

    ``layout``:
    - ``dict`` (default): per-arm ``{name: {flux, ivar?, mask?}}`` (or flat for one arm)
    - ``stack``: flux ``[C, nwave]`` when arms share length
    - ``concat``: one 1D flux with parallel ivar/mask
    """

    def __init__(
        self,
        paths: str | list[str],
        hdu: HduSpec = 0,
        ivar_hdu: HduSpec | None = None,
        mask_hdu: HduSpec | None = None,
        column: str | None = None,
        ivar_column: str | None = None,
        row: int | None = None,
        layout: str = "dict",
        transform: Callable[..., Any] | None = None,
        device: str = "cpu",
        mmap: bool | str = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        if layout not in {"dict", "stack", "concat"}:
            raise ValueError("layout must be 'dict', 'stack', or 'concat'")
        self.files = _resolve_paths(paths)
        self.hdus = _as_hdu_list(hdu)
        self.ivar_hdus = None if ivar_hdu is None else _as_hdu_list(ivar_hdu)
        self.mask_hdus = None if mask_hdu is None else _as_hdu_list(mask_hdu)
        if self.ivar_hdus is not None and len(self.ivar_hdus) != len(self.hdus):
            raise ValueError("ivar_hdu must match hdu arity")
        if self.mask_hdus is not None and len(self.mask_hdus) != len(self.hdus):
            raise ValueError("mask_hdu must match hdu arity")
        self.column = column
        self.ivar_column = ivar_column
        self.row = row
        self.layout = layout
        self.transform = transform
        self.device = device
        self.mmap = mmap
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.hdu = self.hdus[0] if len(self.hdus) == 1 else self.hdus

    def __len__(self) -> int:
        return len(self.files)

    def _to_1d(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.row is not None:
            tensor = tensor[self.row]
        if tensor.ndim > 1:
            # Keep 2D [nspec, nwave] when row is unset; flatten true 1D leftovers.
            if tensor.ndim == 2 and self.row is None:
                return tensor
            return tensor.reshape(-1)
        return tensor

    def _read_table_arm(self, path: str) -> dict[str, torch.Tensor]:
        from torchfits import table as tf_table

        if self.column is None:
            raise ValueError("table spectrum path requires column=")
        hdu = self.hdus[0]
        if not isinstance(hdu, int):
            raise ValueError("table spectrum path requires integer hdu index")
        names = [self.column]
        if self.ivar_column is not None:
            names.append(self.ivar_column)
        cols = tf_table.read_torch(path, hdu=hdu, columns=names, device=self.device)
        flux = self._to_1d(cols[self.column])
        out: dict[str, torch.Tensor] = {"flux": flux}
        if self.ivar_column is not None:
            out["ivar"] = self._to_1d(cols[self.ivar_column])
        return out

    def _read_image_arms(self, path: str) -> list[dict[str, torch.Tensor]]:
        arms: list[dict[str, torch.Tensor]] = []
        for i, hdu in enumerate(self.hdus):
            flux = self._to_1d(
                _read_image(path, hdu, device=self.device, mmap=self.mmap)
            )
            arm: dict[str, torch.Tensor] = {"flux": flux}
            if self.ivar_hdus is not None:
                arm["ivar"] = self._to_1d(
                    _read_image(
                        path, self.ivar_hdus[i], device=self.device, mmap=self.mmap
                    )
                )
            if self.mask_hdus is not None:
                arm["mask"] = self._to_1d(
                    _read_image(
                        path, self.mask_hdus[i], device=self.device, mmap=self.mmap
                    )
                )
            arms.append(arm)
        return arms

    def _layout_arms(self, arms: list[dict[str, torch.Tensor]]) -> Any:
        if self.layout == "dict":
            if len(arms) == 1:
                return arms[0]
            return {_arm_name(self.hdus[i]): arm for i, arm in enumerate(arms)}
        fluxes = [arm["flux"] for arm in arms]
        if self.layout == "stack":
            lengths = {int(f.shape[-1]) for f in fluxes}
            if len(lengths) != 1:
                raise ValueError(
                    "layout='stack' requires equal nwave per arm; "
                    f"got lengths {sorted(lengths)}"
                )
            flux = torch.stack(fluxes, dim=0)
            out: dict[str, torch.Tensor] = {"flux": flux}
            if all("ivar" in arm for arm in arms):
                out["ivar"] = torch.stack([arm["ivar"] for arm in arms], dim=0)
            if all("mask" in arm for arm in arms):
                out["mask"] = torch.stack([arm["mask"] for arm in arms], dim=0)
            return out
        # concat
        flux = torch.cat(fluxes, dim=-1)
        out = {"flux": flux}
        if all("ivar" in arm for arm in arms):
            out["ivar"] = torch.cat([arm["ivar"] for arm in arms], dim=-1)
        if all("mask" in arm for arm in arms):
            out["mask"] = torch.cat([arm["mask"] for arm in arms], dim=-1)
        return out

    def __getitem__(self, idx: int) -> Any:
        ahead = self.files[idx + 1 : idx + 3]
        path = _local_read_path(
            self.files[idx], prefetch_ahead=ahead, cache_dir=self.cache_dir
        )
        if self.column is not None:
            payload = self._read_table_arm(path)
            if self.layout != "dict" and len(self.hdus) > 1:
                raise ValueError("table column spectra only support a single arm")
        else:
            payload = self._layout_arms(self._read_image_arms(path))
        if self.transform is not None:
            payload = self.transform(payload)
        return payload

    def __repr__(self) -> str:
        return (
            f"FitsSpectrumDataset(n={len(self.files)}, hdu={self.hdu!r}, "
            f"layout={self.layout!r}, column={self.column!r})"
        )


class FitsSpectrumIterableDataset(IterableDataset[Any]):
    """1D spectra streaming (IMAGE or table column), multi-arm layouts."""

    def __init__(
        self,
        paths: str | list[str],
        hdu: HduSpec = 0,
        ivar_hdu: HduSpec | None = None,
        mask_hdu: HduSpec | None = None,
        column: str | None = None,
        ivar_column: str | None = None,
        row: int | None = None,
        layout: str = "dict",
        transform: Callable[..., Any] | None = None,
        device: str = "cpu",
        mmap: bool | str = True,
        shuffle: bool = False,
        shuffle_buffer_size: int | None = None,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if layout not in {"dict", "stack", "concat"}:
            raise ValueError("layout must be 'dict', 'stack', or 'concat'")
        self.files = _resolve_paths(paths)
        self.hdus = _as_hdu_list(hdu)
        self.ivar_hdus = None if ivar_hdu is None else _as_hdu_list(ivar_hdu)
        self.mask_hdus = None if mask_hdu is None else _as_hdu_list(mask_hdu)
        if self.ivar_hdus is not None and len(self.ivar_hdus) != len(self.hdus):
            raise ValueError("ivar_hdu must match hdu arity")
        if self.mask_hdus is not None and len(self.mask_hdus) != len(self.hdus):
            raise ValueError("mask_hdu must match hdu arity")
        self.column = column
        self.ivar_column = ivar_column
        self.row = row
        self.layout = layout
        self.transform = transform
        self.device = device
        self.mmap = mmap
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.hdu = self.hdus[0] if len(self.hdus) == 1 else self.hdus
        self._spec_reader = FitsSpectrumDataset(
            self.files[:1],
            hdu=self.hdu,
            ivar_hdu=self.ivar_hdus,
            mask_hdu=self.mask_hdus,
            column=self.column,
            ivar_column=self.ivar_column,
            row=self.row,
            layout=self.layout,
            device=self.device,
            mmap=self.mmap,
        )

    def _generate(self) -> Iterator[Any]:
        rank, world_size = _resolve_rank_and_world_size(self.rank, self.world_size)
        sharded_files = _shard_sequence(self.files, rank, world_size)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            indices = list(range(len(sharded_files)))
            worker_seed = self.seed
        else:
            total = len(sharded_files)
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = total // num_workers
            remainder = total % num_workers
            start = worker_id * per_worker + min(worker_id, remainder)
            size = per_worker + (1 if worker_id < remainder else 0)
            indices = list(range(start, start + size))
            worker_seed = self.seed + worker_id

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(worker_seed)
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]

        for i, idx in enumerate(indices):
            ahead = [
                sharded_files[indices[j]]
                for j in range(i + 1, min(i + 3, len(indices)))
            ]
            path = _local_read_path(
                sharded_files[idx], prefetch_ahead=ahead, cache_dir=self.cache_dir
            )
            if self.column is not None:
                payload = self._spec_reader._read_table_arm(path)
                if self.layout != "dict" and len(self.hdus) > 1:
                    raise ValueError("table column spectra only support a single arm")
            else:
                payload = self._spec_reader._layout_arms(
                    self._spec_reader._read_image_arms(path)
                )
            if self.transform is not None:
                payload = self.transform(payload)
            yield payload

    def __iter__(self) -> Iterator[Any]:
        stream = self._generate()
        if self.shuffle_buffer_size is not None and self.shuffle_buffer_size > 1:
            stream = _buffered_shuffle(
                stream, buffer_size=self.shuffle_buffer_size, seed=self.seed
            )
        return stream

    def __repr__(self) -> str:
        return (
            f"FitsSpectrumIterableDataset(n={len(self.files)}, hdu={self.hdu!r}, "
            f"layout={self.layout!r}, column={self.column!r})"
        )


class FitsStagedCutoutIterableDataset(IterableDataset[Any]):
    """Streaming cutouts from large remote or local FITS mosaics with ephemeral staging."""

    def __init__(
        self,
        paths: str | list[str],
        cutouts_per_file: int = 100,
        cutout_size: int | tuple[int, int] = 128,
        hdu: HduRef = 0,
        *,
        staging_dir: str | Path | None = None,
        cleanup: bool = True,
        cutout_generator: (
            Callable[[int, int, int, int], tuple[int, int, int, int]] | None
        ) = None,
        transform: Callable[..., Any] | None = None,
        device: str = "cpu",
        add_channel_dim: bool = True,
        shuffle_files: bool = False,
        shuffle_buffer_size: int | None = None,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        self.files = _resolve_paths(paths)
        self.cutouts_per_file = max(1, int(cutouts_per_file))
        if isinstance(cutout_size, int):
            self.cutout_size = (cutout_size, cutout_size)
        else:
            self.cutout_size = (int(cutout_size[0]), int(cutout_size[1]))
        self.hdu = hdu
        self.staging_dir = Path(staging_dir) if staging_dir is not None else None
        self.cleanup = cleanup
        self.cutout_generator = cutout_generator
        self.transform = transform
        self.device = device
        self.add_channel_dim = add_channel_dim
        self.shuffle_files = shuffle_files
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

    def _default_cutout_coords(
        self, height: int, width: int, ch: int, cw: int, rng: Any
    ) -> tuple[int, int, int, int]:
        max_y = max(0, height - ch)
        max_x = max(0, width - cw)
        y1 = rng.randint(0, max_y) if max_y > 0 else 0
        x1 = rng.randint(0, max_x) if max_x > 0 else 0
        return x1, y1, min(width, x1 + cw), min(height, y1 + ch)

    def _generate(self) -> Iterator[Any]:
        import random

        from torchfits.io import open_subset_reader

        from .remote import (
            cleanup_downloaded_file,
            ephemeral_scratch_dir,
            is_remote_url,
        )

        stage_root = self.staging_dir or ephemeral_scratch_dir()
        rank, world_size = _resolve_rank_and_world_size(self.rank, self.world_size)
        sharded_files = _shard_sequence(self.files, rank, world_size)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            indices = list(range(len(sharded_files)))
            worker_seed = self.seed
        else:
            total = len(sharded_files)
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = total // num_workers
            remainder = total % num_workers
            start = worker_id * per_worker + min(worker_id, remainder)
            size = per_worker + (1 if worker_id < remainder else 0)
            indices = list(range(start, start + size))
            worker_seed = self.seed + worker_id

        if self.shuffle_files:
            g = torch.Generator()
            g.manual_seed(worker_seed)
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]

        rng = random.Random(worker_seed)
        ch, cw = self.cutout_size

        for i, idx in enumerate(indices):
            file_ref = sharded_files[idx]
            is_remote = is_remote_url(file_ref)
            ahead = [
                sharded_files[indices[j]]
                for j in range(i + 1, min(i + 3, len(indices)))
            ]
            local_path = _local_read_path(
                file_ref, prefetch_ahead=ahead, cache_dir=stage_root
            )

            try:
                with open_subset_reader(
                    local_path, hdu=self.hdu, device=self.device
                ) as reader:
                    height, width = reader.shape
                    for _ in range(self.cutouts_per_file):
                        if self.cutout_generator is not None:
                            x1, y1, x2, y2 = self.cutout_generator(
                                height, width, ch, cw
                            )
                        else:
                            x1, y1, x2, y2 = self._default_cutout_coords(
                                height, width, ch, cw, rng
                            )
                        cutout = reader.read_subset(x1, y1, x2, y2)
                        if self.add_channel_dim and cutout.ndim == 2:
                            cutout = cutout.unsqueeze(0)
                        if self.transform is not None:
                            cutout = self.transform(cutout)
                        yield cutout
            finally:
                if self.cleanup and is_remote:
                    cleanup_downloaded_file(local_path)

    def __iter__(self) -> Iterator[Any]:
        stream = self._generate()
        if self.shuffle_buffer_size is not None and self.shuffle_buffer_size > 1:
            stream = _buffered_shuffle(
                stream, buffer_size=self.shuffle_buffer_size, seed=self.seed
            )
        return stream

    def __repr__(self) -> str:
        return (
            f"FitsStagedCutoutIterableDataset(n_files={len(self.files)}, "
            f"cutouts_per_file={self.cutouts_per_file}, cutout_size={self.cutout_size})"
        )
