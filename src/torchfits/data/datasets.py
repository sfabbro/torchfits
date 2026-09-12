"""Map/iterable Datasets for IMAGE tensors, images, cubes, and spectra."""

from __future__ import annotations

import contextlib
import glob as _glob
import os
import random
from dataclasses import dataclass
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


def _worker_shard(
    seq: list[Any], rank: int, world_size: int, seed: int
) -> tuple[list[Any], list[int], int]:
    """Shard ``seq`` by rank, then by DataLoader worker.

    Returns ``(sharded, indices, worker_seed)`` where ``indices`` is this
    worker's contiguous slice of ``sharded`` (leftover rows go to the
    first workers) and ``worker_seed`` is ``seed + worker_id``.
    """
    sharded = _shard_sequence(seq, rank, world_size)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return sharded, list(range(len(sharded))), seed
    total = len(sharded)
    num_workers = worker_info.num_workers
    worker_id = worker_info.id
    per_worker = total // num_workers
    remainder = total % num_workers
    start = worker_id * per_worker + min(worker_id, remainder)
    size = per_worker + (1 if worker_id < remainder else 0)
    return sharded, list(range(start, start + size)), seed + worker_id


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


def _resolve_file_labels(
    files: list[str],
    *,
    label_key: str | None,
    labels: list[int] | None,
    hdu: int = 0,
) -> list[int] | None:
    """Per-file integer labels from an explicit list or a primary-header key.

    Returns ``None`` when neither is supplied, which keeps the legacy
    label-free payload return for spectra.
    """
    if labels is not None:
        if len(labels) != len(files):
            raise ValueError(
                f"labels length {len(labels)} != files length {len(files)}"
            )
        return [int(v) for v in labels]
    if label_key is None:
        return None
    from torchfits import read_keys

    return [int(read_keys(path, [label_key], hdu=hdu)[label_key]) for path in files]


# Zeropoint header keys, most specific first. FITS files name the same
# quantity differently across surveys / instruments.
_ZEROPOINT_KEYS = (
    "PHOTZEROPOINT",
    "PHOTZP",
    "ZP",
    "MAGZERO",
    "MAGZP",
    "ABMAGZERO",
    "ZEROPOINT",
)
_IVAR_SUFFIXES = ("_IVAR", "IVAR")
_MASK_SUFFIXES = ("_MASK", "_DQ", "MASK", "DQ")
_DQ_SUFFIXES = ("_DQ", "DQ")
_WAVELENGTH_SUFFIXES = ("_WAVELENGTH", "_WAVE", "WAVELENGTH", "WAVE")
_BITPIX_DTYPES = {
    8: "uint8",
    16: "int16",
    32: "int32",
    64: "int64",
    -32: "float32",
    -64: "float64",
}


@dataclass(frozen=True)
class BandInfo:
    """One image extension found by :func:`discover_bands`.

    ``zeropoint`` follows the ``mag = -2.5 log10(counts) + ZP`` convention so
    that ``flux_scale()`` converts stored counts to physical flux.
    """

    index: int
    name: str
    shape: tuple[int, ...]
    dtype: str
    role: str = "flux"  # "flux" | "ivar" | "mask" | "wavelength"
    bitpix: int | None = None
    zeropoint: float | None = None
    photflam: float | None = None
    photplam: float | None = None
    exptime: float | None = None

    def flux_scale(self, *, exptime_normalized: bool = True) -> float | None:
        """Factor turning counts into flux: ``10**(-0.4 * ZP)``.

        Returns ``None`` when the header declares no zeropoint. With
        ``exptime_normalized=False`` the factor is divided by ``EXPTIME`` so
        the result is per-second.
        """
        if self.zeropoint is None:
            return None
        scale = float(10.0 ** (-0.4 * float(self.zeropoint)))
        if not exptime_normalized and self.exptime:
            scale /= float(self.exptime)
        return scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "role": self.role,
            "bitpix": self.bitpix,
            "zeropoint": self.zeropoint,
            "photflam": self.photflam,
            "photplam": self.photplam,
            "exptime": self.exptime,
        }


def _band_role(name: str) -> str:
    """Classify an image extension by its companion-style name suffix."""
    upper = name.upper()
    for suffixes, role in (
        (_IVAR_SUFFIXES, "ivar"),
        (_MASK_SUFFIXES, "mask"),
        (_WAVELENGTH_SUFFIXES, "wavelength"),
    ):
        if any(upper.endswith(suffix) for suffix in suffixes):
            return role
    return "flux"


def discover_bands(
    path: str | Path,
    *,
    extra_keys: Sequence[str] = (),
) -> list[BandInfo]:
    """List the 2D+ image extensions of *path* with photometric metadata.

    Lets a multi-band dataset be built from names (``"G"``, ``"R"``, ``"Z"``)
    instead of hand-written HDU indices, and exposes the zeropoints so counts
    can be turned into physical flux with :meth:`BandInfo.flux_scale`.

    Parameters
    ----------
    path :
        Local path or remote URL for a FITS file.
    extra_keys :
        Additional header keywords to surface. Currently used only to pick a
        per-band ``FILTER``/``BAND`` label when ``EXTNAME`` is missing.

    Returns
    -------
    list[BandInfo]
        Image extensions in file order; table HDUs and 1D extensions are
        skipped.
    """
    import torchfits

    resolved = resolve_local_path(str(path))
    n_hdus = int(torchfits.read_num_hdus(resolved))
    label_keys = ("FILTER", "BAND", *extra_keys)
    bands: list[BandInfo] = []
    for index in range(n_hdus):
        try:
            bitpix, shape = torchfits.read_shape(resolved, index)
        except Exception:
            continue
        if len(shape) < 2:
            continue
        try:
            header = torchfits.read_header(resolved, index)
        except Exception:
            continue
        extname = header.get("EXTNAME")
        name = str(extname).strip() if extname else f"HDU{index}"
        if not extname:
            for key in label_keys:
                value = header.get(key)
                if value:
                    name = str(value).strip()
                    break
        zeropoint = None
        for key in _ZEROPOINT_KEYS:
            value = header.get(key)
            if value is not None:
                try:
                    zeropoint = float(value)
                except (TypeError, ValueError):
                    continue
                break

        def _float(key: str) -> float | None:
            value = header.get(key)
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        bands.append(
            BandInfo(
                index=index,
                name=name,
                shape=tuple(int(s) for s in shape),
                dtype=_BITPIX_DTYPES.get(int(bitpix), f"BITPIX{int(bitpix)}"),
                role=_band_role(name),
                bitpix=int(bitpix),
                zeropoint=zeropoint,
                photflam=_float("PHOTFLAM"),
                photplam=_float("PHOTPLAM"),
                exptime=_float("EXPTIME"),
            )
        )
    return bands


def _slice_leading(
    payload: Any, *, index: int | None, window: tuple[int, int] | None
) -> Any:
    """Index and/or window the leading (spectral) axis of a cube payload."""
    if index is None and window is None:
        return payload

    def _apply(tensor: torch.Tensor) -> torch.Tensor:
        if index is not None:
            tensor = tensor[index]
        if window is not None:
            tensor = tensor[window[0] : window[1]]
        return tensor

    if isinstance(payload, dict):
        return {
            key: _apply(value) if torch.is_tensor(value) else value
            for key, value in payload.items()
        }
    return _apply(payload)


def _pick_hdu(hdus: list[HduRef], index: int) -> HduRef:
    """Companion HDUs may be a single shared extension or one per arm."""
    return hdus[index] if len(hdus) > 1 else hdus[0]


def _as_validity_mask(
    tensor: torch.Tensor, *, is_dq: bool, bad_bits: Any
) -> torch.Tensor:
    """Boolean ``True`` = valid. Integer DQ bitfields go through ``mask_from_dq``."""
    if not is_dq:
        return tensor.to(torch.bool)
    from torchfits.transforms import mask_from_dq

    return mask_from_dq(tensor, bad_bits=bad_bits)


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
    mask_is_dq: bool = False,
    bad_bits: Any = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    flux = _read_flux_stack(path, hdus, device=device, mmap=mmap)
    if add_channel_dim and flux.ndim == 2:
        flux = flux.unsqueeze(0)
    ivar = _optional_companion(path, ivar_hdus, device=device, mmap=mmap)
    mask = _optional_companion(path, mask_hdus, device=device, mmap=mmap)
    if add_channel_dim:
        # A single companion HDU has no channel axis, so give it the same one
        # the flux got.  Without this ``payload["mask"][0]`` on a one-band
        # image silently means "row 0" instead of "channel 0".
        if ivar is not None and ivar.ndim == 2:
            ivar = ivar.unsqueeze(0)
        if mask is not None and mask.ndim == 2:
            mask = mask.unsqueeze(0)
    if mask is not None:
        # A companion ``*_MASK``/``*_DQ`` extension is delivered as a boolean
        # *validity* mask (``True`` = valid) — the one convention every
        # transform and statistic in torchfits shares.  Raw DQ bitfields must
        # be decoded (``mask_is_dq=True``); a perfectly clean, all-zero DQ
        # frame handed to the stats as an integer "mask" would mark *every*
        # pixel invalid and turn each reduction into NaN.
        mask = _as_validity_mask(mask, is_dq=mask_is_dq, bad_bits=bad_bits)
    payload = _pack_payload(flux, ivar, mask)
    if transform is not None:
        payload = transform(payload)
    return payload


class FitsTensorDataset(Dataset[Any]):
    """General N-D IMAGE HDU → tensor (any rank).

    Multi-HDU ``hdu=[…]`` stacks **flux** channels on dim 0. Optional
    ``ivar_hdu`` / ``mask_hdu`` are companion tensors (never flux channels).

    A ``mask_hdu`` companion is returned under the ``mask`` key as a
    **boolean validity** mask (``True`` = valid), the convention every
    transform and statistic in torchfits uses. Set ``mask_is_dq=True`` for an
    integer FITS ``DQ`` bitfield so it is decoded through
    :func:`~torchfits.transforms.mask_from_dq` (name the fatal bits with
    ``bad_bits=``) rather than being reinterpreted as a validity array.
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
        *,
        mask_is_dq: bool = False,
        bad_bits: int | Sequence[int] | None = None,
    ) -> None:
        self.files = _resolve_paths(paths)
        self.hdus = _as_hdu_list(hdu)
        self.ivar_hdus = None if ivar_hdu is None else _as_hdu_list(ivar_hdu)
        self.mask_hdus = None if mask_hdu is None else _as_hdu_list(mask_hdu)
        if self.ivar_hdus is not None and len(self.ivar_hdus) != len(self.hdus):
            raise ValueError("ivar_hdu must match hdu arity")
        if self.mask_hdus is not None and len(self.mask_hdus) != len(self.hdus):
            raise ValueError("mask_hdu must match hdu arity")
        self.mask_is_dq = bool(mask_is_dq)
        self.bad_bits = bad_bits
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
            mask_is_dq=self.mask_is_dq,
            bad_bits=self.bad_bits,
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

    @classmethod
    def from_bands(
        cls,
        paths: str | list[str],
        bands: Sequence[str] | None = None,
        *,
        ivar_suffixes: Sequence[str] = _IVAR_SUFFIXES,
        mask_suffixes: Sequence[str] = _MASK_SUFFIXES,
        mask_is_dq: bool | None = None,
        bad_bits: int | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> "FitsImageDataset":
        """Build a multi-band dataset by discovering image extensions by name.

        ``bands=None`` selects every extension whose name does not look like a
        companion (``G``, ``R``, ``Z`` — not ``G_IVAR`` / ``G_DQ``). Companion
        IVAR / mask extensions are attached only when *every* selected band has
        one, matched by suffix (``"G"`` → ``"G_IVAR"``, ``"G_MASK"/"G_DQ"``).
        Band names come from ``EXTNAME``, falling back to
        ``FILTER``/``BAND``/``HDU<n>``.

        A ``*_DQ`` companion is decoded as a FITS bitfield through
        :func:`~torchfits.transforms.mask_from_dq` (suffix-derived, so it needs
        no extra arguments); pass ``mask_is_dq=False`` to override, or
        ``bad_bits=`` to name the fatal bits. Either way the payload's
        ``mask`` is a boolean validity mask where ``True`` means valid.

        Use :func:`discover_bands` to inspect zeropoints before choosing
        bands; see :meth:`BandInfo.flux_scale` for counts→flux conversion.
        """
        files = _resolve_paths(paths)
        if not files:
            raise ValueError("no FITS paths matched")
        infos = discover_bands(files[0])
        available = [info.name for info in infos]
        if bands is None:
            selected = [info.name for info in infos if info.role == "flux"]
            if not selected:
                selected = available
        else:
            selected = list(bands)
            missing = [b for b in selected if b not in set(available)]
            if missing:
                raise ValueError(
                    f"bands not found in {files[0]!r}: {missing}; "
                    f"available: {available}"
                )
        if not selected:
            raise ValueError(f"no image bands found in {files[0]!r}")

        known = set(available)

        def _companion(band: str, suffixes: Sequence[str]) -> tuple[str, str] | None:
            for suffix in suffixes:
                if f"{band}{suffix}" in known:
                    return f"{band}{suffix}", suffix
            return None

        ivar_hits = [
            hit for hit in (_companion(b, ivar_suffixes) for b in selected) if hit
        ]
        mask_hits = [
            hit for hit in (_companion(b, mask_suffixes) for b in selected) if hit
        ]
        attach_ivar = len(ivar_hits) == len(selected)
        attach_mask = len(mask_hits) == len(selected)

        # A ``*_DQ`` extension is a bitfield, not a validity mask. Decide from
        # the suffixes so the caller does not have to: mixing the two kinds
        # across bands is ambiguous, so say so instead of guessing.
        detected_is_dq: bool | None = None
        if attach_mask:
            kinds = {suffix in _DQ_SUFFIXES for _, suffix in mask_hits}
            if len(kinds) > 1:
                raise ValueError(
                    "mask companions mix DQ bitfields and plain masks across "
                    "bands; build the dataset with explicit mask_hdu=/"
                    "mask_is_dq= instead"
                )
            detected_is_dq = kinds.pop()

        return cls(
            files,
            hdu=selected,
            ivar_hdu=[name for name, _ in ivar_hits] if attach_ivar else None,
            mask_hdu=[name for name, _ in mask_hits] if attach_mask else None,
            mask_is_dq=detected_is_dq if mask_is_dq is None else mask_is_dq,
            bad_bits=bad_bits,
            **kwargs,
        )

    def band_zeropoints(self) -> dict[str, float | None]:
        """``{band_name: zeropoint}`` for this dataset's flux HDUs.

        Requires the flux HDUs to be named extensions (the ``from_bands``
        path); unnamed integer HDUs are reported under ``"HDU<n>"``.
        """
        if not self.files:
            return {}
        return {
            info.name: info.zeropoint
            for info in discover_bands(self.files[0])
            if info.role == "flux"
        }

    def __repr__(self) -> str:
        return (
            f"FitsImageDataset(n={len(self.files)}, hdu={self.hdu!r}, "
            f"device={self.device!r})"
        )


class FitsCubeDataset(FitsTensorDataset):
    """3D+ datacube peer (IFU / velocity / radio cubes).

    The leading axis is the spectral (or velocity/channel) axis. Pick a single
    channel with ``slice_index=`` or a contiguous band/IFU window with
    ``spectral_slice=(start, stop)`` — the latter is what most IFU pipelines
    want (e.g. a rest-wavelength range instead of the full cube). The two are
    mutually exclusive.
    """

    def __init__(
        self,
        paths: str | list[str],
        hdu: HduSpec = 0,
        slice_index: int | None = None,
        spectral_slice: tuple[int, int] | None = None,
        *,
        add_channel_dim: bool = False,
        **kwargs: Any,
    ) -> None:
        if slice_index is not None and spectral_slice is not None:
            raise ValueError("pass slice_index= or spectral_slice=, not both")
        if spectral_slice is not None:
            start, stop = int(spectral_slice[0]), int(spectral_slice[1])
            if start < 0 or stop <= start:
                raise ValueError(
                    "spectral_slice must be (start, stop) with 0 <= start < stop"
                )
            self.spectral_slice: tuple[int, int] | None = (start, stop)
        else:
            self.spectral_slice = None
        self.slice_index = slice_index
        super().__init__(paths, hdu=hdu, add_channel_dim=add_channel_dim, **kwargs)

    def __getitem__(self, idx: int) -> tuple[Any, torch.Tensor]:
        payload, label = super().__getitem__(idx)
        return (
            _slice_leading(payload, index=self.slice_index, window=self.spectral_slice),
            label,
        )

    def __repr__(self) -> str:
        return (
            f"FitsCubeDataset(n={len(self.files)}, hdu={self.hdu!r}, "
            f"slice_index={self.slice_index!r}, "
            f"spectral_slice={self.spectral_slice!r})"
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
        *,
        mask_is_dq: bool = False,
        bad_bits: int | Sequence[int] | None = None,
    ) -> None:
        self.files = _resolve_paths(paths)
        self.hdus = _as_hdu_list(hdu)
        self.ivar_hdus = None if ivar_hdu is None else _as_hdu_list(ivar_hdu)
        self.mask_hdus = None if mask_hdu is None else _as_hdu_list(mask_hdu)
        self.mask_is_dq = bool(mask_is_dq)
        self.bad_bits = bad_bits
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
        sharded_files, indices, worker_seed = _worker_shard(
            self.files, rank, world_size, self.seed
        )

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
                mask_is_dq=self.mask_is_dq,
                bad_bits=self.bad_bits,
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
    """3D+ datacube streaming peer for IFU / velocity / radio cubes.

    Same leading-axis selection as :class:`FitsCubeDataset`:
    ``slice_index=`` for one channel, ``spectral_slice=(start, stop)`` for a
    contiguous spectral window (mutually exclusive).
    """

    def __init__(
        self,
        paths: str | list[str],
        hdu: HduSpec = 0,
        slice_index: int | None = None,
        spectral_slice: tuple[int, int] | None = None,
        *,
        add_channel_dim: bool = False,
        **kwargs: Any,
    ) -> None:
        if slice_index is not None and spectral_slice is not None:
            raise ValueError("pass slice_index= or spectral_slice=, not both")
        if spectral_slice is not None:
            start, stop = int(spectral_slice[0]), int(spectral_slice[1])
            if start < 0 or stop <= start:
                raise ValueError(
                    "spectral_slice must be (start, stop) with 0 <= start < stop"
                )
            self.spectral_slice: tuple[int, int] | None = (start, stop)
        else:
            self.spectral_slice = None
        self.slice_index = slice_index
        super().__init__(paths, hdu=hdu, add_channel_dim=add_channel_dim, **kwargs)

    def __iter__(self) -> Iterator[Any]:
        for payload in super().__iter__():
            yield _slice_leading(
                payload, index=self.slice_index, window=self.spectral_slice
            )

    def __repr__(self) -> str:
        return (
            f"FitsCubeIterableDataset(n={len(self.files)}, hdu={self.hdu!r}, "
            f"slice_index={self.slice_index!r}, "
            f"spectral_slice={self.spectral_slice!r})"
        )


class FitsSpectrumDataset(Dataset[Any]):
    """1D spectra (IMAGE or table column), optional multi-arm layouts.

    ``layout``:
    - ``dict`` (default): per-arm ``{name: {flux, ivar?, mask?, wavelength?}}``
      (flat keys for a single arm)
    - ``stack``: flux ``[C, nwave]`` when arms share length
    - ``concat``: one 1D flux with parallel ivar/mask/wavelength

    Optional ``wavelength_hdu`` / ``wavelength_column`` attach a parallel
    ``wavelength`` tensor so spectra can be resampled, rest-framed or
    interpolated downstream. ``mask_hdu`` / ``mask_column`` attach a validity
    mask; with ``mask_is_dq=True`` the integer extension is interpreted as a
    FITS ``DQ`` bitfield through :func:`torchfits.transforms.mask_from_dq`
    (see ``bad_bits``) instead of being used verbatim.

    Returns the payload alone, or ``(payload, label)`` when ``labels=`` or
    ``label_key=`` is supplied.
    """

    def __init__(
        self,
        paths: str | list[str],
        hdu: HduSpec = 0,
        ivar_hdu: HduSpec | None = None,
        mask_hdu: HduSpec | None = None,
        column: str | None = None,
        ivar_column: str | None = None,
        mask_column: str | None = None,
        wavelength_column: str | None = None,
        wavelength_hdu: HduSpec | None = None,
        row: int | None = None,
        layout: str = "dict",
        label_key: str | None = None,
        labels: list[int] | None = None,
        transform: Callable[..., Any] | None = None,
        device: str = "cpu",
        mmap: bool | str = True,
        cache_dir: str | Path | None = None,
        *,
        mask_is_dq: bool = False,
        bad_bits: int | Sequence[int] | None = None,
    ) -> None:
        if layout not in {"dict", "stack", "concat"}:
            raise ValueError("layout must be 'dict', 'stack', or 'concat'")
        if mask_is_dq and mask_hdu is None and mask_column is None:
            raise ValueError("mask_is_dq=True requires mask_hdu= or mask_column=")
        if column is not None and (wavelength_hdu is not None or mask_hdu is not None):
            raise ValueError(
                "table spectra read companions from columns; "
                "use wavelength_column=/mask_column= instead of *_hdu="
            )
        self.files = _resolve_paths(paths)
        self.hdus = _as_hdu_list(hdu)
        self.ivar_hdus = None if ivar_hdu is None else _as_hdu_list(ivar_hdu)
        self.mask_hdus = None if mask_hdu is None else _as_hdu_list(mask_hdu)
        if self.ivar_hdus is not None and len(self.ivar_hdus) != len(self.hdus):
            raise ValueError("ivar_hdu must match hdu arity")
        self.wavelength_hdus = (
            None if wavelength_hdu is None else _as_hdu_list(wavelength_hdu)
        )
        if self.wavelength_hdus is not None and len(self.wavelength_hdus) not in (
            1,
            len(self.hdus),
        ):
            raise ValueError("wavelength_hdu must be a single HDU or match hdu arity")
        if self.mask_hdus is not None and len(self.mask_hdus) not in (
            1,
            len(self.hdus),
        ):
            raise ValueError("mask_hdu must be a single HDU or match hdu arity")
        self.column = column
        self.ivar_column = ivar_column
        self.mask_column = mask_column
        self.wavelength_column = wavelength_column
        self.row = row
        self.layout = layout
        self.transform = transform
        self.device = device
        self.mmap = mmap
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.hdu = self.hdus[0] if len(self.hdus) == 1 else self.hdus
        self.mask_is_dq = bool(mask_is_dq)
        self.bad_bits = bad_bits
        resolved = _resolve_file_labels(self.files, label_key=label_key, labels=labels)
        self.labels = resolved

    def __len__(self) -> int:
        return len(self.files)

    def _to_1d(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.row is not None:
            if tensor.ndim < 2:
                raise ValueError(
                    f"row={self.row} needs a multi-row HDU/column of rank >= 2, "
                    f"but this one has shape {tuple(tensor.shape)}. Drop row= "
                    "to read it as a single spectrum."
                )
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
        for extra in (
            self.ivar_column,
            self.mask_column,
            self.wavelength_column,
        ):
            if extra is not None:
                names.append(extra)
        cols = tf_table.read_torch(path, hdu=hdu, columns=names, device=self.device)
        out: dict[str, torch.Tensor] = {"flux": self._to_1d(cols[self.column])}
        if self.ivar_column is not None:
            out["ivar"] = self._to_1d(cols[self.ivar_column])
        if self.mask_column is not None:
            out["mask"] = _as_validity_mask(
                self._to_1d(cols[self.mask_column]),
                is_dq=self.mask_is_dq,
                bad_bits=self.bad_bits,
            )
        if self.wavelength_column is not None:
            out["wavelength"] = self._to_1d(cols[self.wavelength_column])
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
                        path,
                        _pick_hdu(self.ivar_hdus, i),
                        device=self.device,
                        mmap=self.mmap,
                    )
                )
            if self.mask_hdus is not None:
                arm["mask"] = _as_validity_mask(
                    self._to_1d(
                        _read_image(
                            path,
                            _pick_hdu(self.mask_hdus, i),
                            device=self.device,
                            mmap=self.mmap,
                        )
                    ),
                    is_dq=self.mask_is_dq,
                    bad_bits=self.bad_bits,
                )
            if self.wavelength_hdus is not None:
                arm["wavelength"] = self._to_1d(
                    _read_image(
                        path,
                        _pick_hdu(self.wavelength_hdus, i),
                        device=self.device,
                        mmap=self.mmap,
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
        extra_fields = ("ivar", "mask", "wavelength")
        if self.layout == "stack":
            lengths = {int(f.shape[-1]) for f in fluxes}
            if len(lengths) != 1:
                raise ValueError(
                    "layout='stack' requires equal nwave per arm; "
                    f"got lengths {sorted(lengths)}"
                )
            flux = torch.stack(fluxes, dim=0)
            out: dict[str, torch.Tensor] = {"flux": flux}
            for field in extra_fields:
                if all(field in arm for arm in arms):
                    out[field] = torch.stack([arm[field] for arm in arms], dim=0)
            return out
        # concat
        flux = torch.cat(fluxes, dim=-1)
        out = {"flux": flux}
        for field in extra_fields:
            if all(field in arm for arm in arms):
                out[field] = torch.cat([arm[field] for arm in arms], dim=-1)
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
        if self.labels is None:
            return payload
        return payload, torch.tensor(self.labels[idx], dtype=torch.long)

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
        mask_column: str | None = None,
        wavelength_column: str | None = None,
        wavelength_hdu: HduSpec | None = None,
        row: int | None = None,
        layout: str = "dict",
        label_key: str | None = None,
        labels: list[int] | None = None,
        transform: Callable[..., Any] | None = None,
        device: str = "cpu",
        mmap: bool | str = True,
        shuffle: bool = False,
        shuffle_buffer_size: int | None = None,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
        cache_dir: str | Path | None = None,
        *,
        mask_is_dq: bool = False,
        bad_bits: int | Sequence[int] | None = None,
    ) -> None:
        if layout not in {"dict", "stack", "concat"}:
            raise ValueError("layout must be 'dict', 'stack', or 'concat'")
        self.files = _resolve_paths(paths)
        self.hdus = _as_hdu_list(hdu)
        self.ivar_hdus = None if ivar_hdu is None else _as_hdu_list(ivar_hdu)
        self.mask_hdus = None if mask_hdu is None else _as_hdu_list(mask_hdu)
        if self.ivar_hdus is not None and len(self.ivar_hdus) != len(self.hdus):
            raise ValueError("ivar_hdu must match hdu arity")
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
        resolved = _resolve_file_labels(self.files, label_key=label_key, labels=labels)
        self.labels = resolved
        self._label_by_path = (
            None if resolved is None else dict(zip(self.files, resolved))
        )
        self._spec_reader = FitsSpectrumDataset(
            self.files[:1],
            hdu=self.hdu,
            ivar_hdu=self.ivar_hdus,
            mask_hdu=self.mask_hdus,
            column=self.column,
            ivar_column=self.ivar_column,
            mask_column=mask_column,
            wavelength_column=wavelength_column,
            wavelength_hdu=wavelength_hdu,
            row=self.row,
            layout=self.layout,
            device=self.device,
            mmap=self.mmap,
            mask_is_dq=mask_is_dq,
            bad_bits=bad_bits,
        )

    def _generate(self) -> Iterator[Any]:
        rank, world_size = _resolve_rank_and_world_size(self.rank, self.world_size)
        sharded_files, indices, worker_seed = _worker_shard(
            self.files, rank, world_size, self.seed
        )

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
            if self._label_by_path is None:
                yield payload
            else:
                label = self._label_by_path[sharded_files[idx]]
                yield payload, torch.tensor(label, dtype=torch.long)

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
        hdu: HduSpec = 0,
        ivar_hdu: HduSpec | None = None,
        mask_hdu: HduSpec | None = None,
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
        mask_is_dq: bool = False,
        bad_bits: int | Sequence[int] | None = None,
    ) -> None:
        self.files = _resolve_paths(paths)
        self.cutouts_per_file = max(1, int(cutouts_per_file))
        if isinstance(cutout_size, int):
            self.cutout_size = (cutout_size, cutout_size)
        else:
            self.cutout_size = (int(cutout_size[0]), int(cutout_size[1]))
        self.hdus = _as_hdu_list(hdu)
        self.ivar_hdus = None if ivar_hdu is None else _as_hdu_list(ivar_hdu)
        self.mask_hdus = None if mask_hdu is None else _as_hdu_list(mask_hdu)
        if self.ivar_hdus is not None and len(self.ivar_hdus) != len(self.hdus):
            raise ValueError("ivar_hdu must match hdu arity")
        if self.mask_hdus is not None and len(self.mask_hdus) != len(self.hdus):
            raise ValueError("mask_hdu must match hdu arity")
        self.mask_is_dq = bool(mask_is_dq)
        self.bad_bits = bad_bits
        self.hdu = self.hdus[0] if len(self.hdus) == 1 else self.hdus
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
        sharded_files, indices, worker_seed = _worker_shard(
            self.files, rank, world_size, self.seed
        )

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
                with contextlib.ExitStack() as stack:
                    flux_readers = [
                        stack.enter_context(
                            open_subset_reader(local_path, hdu=h, device=self.device)
                        )
                        for h in self.hdus
                    ]
                    ivar_readers = (
                        [
                            stack.enter_context(
                                open_subset_reader(
                                    local_path, hdu=h, device=self.device
                                )
                            )
                            for h in self.ivar_hdus
                        ]
                        if self.ivar_hdus is not None
                        else []
                    )
                    mask_readers = (
                        [
                            stack.enter_context(
                                open_subset_reader(
                                    local_path, hdu=h, device=self.device
                                )
                            )
                            for h in self.mask_hdus
                        ]
                        if self.mask_hdus is not None
                        else []
                    )

                    height, width = flux_readers[0].shape
                    for _ in range(self.cutouts_per_file):
                        if self.cutout_generator is not None:
                            x1, y1, x2, y2 = self.cutout_generator(
                                height, width, ch, cw
                            )
                        else:
                            x1, y1, x2, y2 = self._default_cutout_coords(
                                height, width, ch, cw, rng
                            )

                        flux_cuts = [
                            r.read_subset(x1, y1, x2, y2) for r in flux_readers
                        ]
                        if len(flux_cuts) == 1:
                            flux = (
                                flux_cuts[0].unsqueeze(0)
                                if self.add_channel_dim and flux_cuts[0].ndim == 2
                                else flux_cuts[0]
                            )
                        else:
                            flux = torch.stack(flux_cuts, dim=0)

                        if self.ivar_hdus is not None:
                            ivar_cuts = [
                                r.read_subset(x1, y1, x2, y2) for r in ivar_readers
                            ]
                            ivar = (
                                torch.stack(ivar_cuts, dim=0)
                                if len(ivar_cuts) > 1
                                else (
                                    ivar_cuts[0].unsqueeze(0)
                                    if self.add_channel_dim and ivar_cuts[0].ndim == 2
                                    else ivar_cuts[0]
                                )
                            )
                        else:
                            ivar = None

                        if self.mask_hdus is not None:
                            mask_cuts = [
                                r.read_subset(x1, y1, x2, y2) for r in mask_readers
                            ]
                            mask = (
                                torch.stack(mask_cuts, dim=0)
                                if len(mask_cuts) > 1
                                else (
                                    mask_cuts[0].unsqueeze(0)
                                    if self.add_channel_dim and mask_cuts[0].ndim == 2
                                    else mask_cuts[0]
                                )
                            )
                        else:
                            mask = None
                        if mask is not None:
                            # Same validity convention as the map-style readers.
                            mask = _as_validity_mask(
                                mask, is_dq=self.mask_is_dq, bad_bits=self.bad_bits
                            )

                        if ivar is not None or mask is not None:
                            payload: Any = {"flux": flux}
                            if ivar is not None:
                                payload["ivar"] = ivar
                            if mask is not None:
                                payload["mask"] = mask
                        else:
                            payload = flux

                        if self.transform is not None:
                            payload = self.transform(payload)
                        yield payload
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
