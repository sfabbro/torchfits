"""
ML data loading throughput benchmark.

Compares torchfits vs fitsio for high-throughput image loading while reporting
repeat-based results and minimizing order bias.
"""

import argparse
import os
import shutil
import tempfile
import time
from statistics import mean, median
from typing import Callable, Dict, List, Tuple

import fitsio
import numpy as np
import torch
import torchfits
from torch.utils.data import DataLoader, Dataset


def create_synthetic_dataset(
    n_files: int = 50, shape: Tuple[int, int] = (512, 512), dtype=np.float32
) -> Tuple[str, List[str]]:
    """Create synthetic uncompressed FITS files."""
    tmp_dir = tempfile.mkdtemp()
    print(f"creating {n_files} synthetic fits files ({shape}) in {tmp_dir}...")

    data = np.random.normal(0, 1, shape).astype(dtype)
    filepaths = []
    for i in range(n_files):
        path = os.path.join(tmp_dir, f"img_{i:04d}.fits")
        fitsio.write(path, data, clobber=True)
        filepaths.append(path)
    return tmp_dir, filepaths


def create_compressed_dataset(
    n_files: int = 50, shape: Tuple[int, int] = (512, 512), dtype=np.float32
) -> Tuple[str, List[str]]:
    """Create synthetic compressed FITS files (Rice)."""
    tmp_dir = tempfile.mkdtemp()
    print(f"creating {n_files} compressed (rice) fits files ({shape}) in {tmp_dir}...")

    data = np.random.normal(0, 1, shape).astype(dtype)
    filepaths = []
    for i in range(n_files):
        path = os.path.join(tmp_dir, f"img_comp_{i:04d}.fits")
        fitsio.write(path, data, clobber=True, compress="rice")
        filepaths.append(path)
    return tmp_dir, filepaths


class TorchFitsDataset(Dataset):
    def __init__(
        self,
        filepaths: List[str],
        *,
        hdu: int | str | None = "auto",
        mmap: bool | str = "auto",
        cache_capacity: int = 0,
        handle_cache_capacity: int = 64,
        scale_on_device: bool = True,
    ):
        self.filepaths = filepaths
        self.hdu = hdu
        self.mmap = mmap
        self.cache_capacity = cache_capacity
        self.handle_cache_capacity = handle_cache_capacity
        self.scale_on_device = scale_on_device

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        return torchfits.read(
            self.filepaths[idx],
            hdu=self.hdu,
            mmap=self.mmap,
            cache_capacity=self.cache_capacity,
            handle_cache_capacity=self.handle_cache_capacity,
            scale_on_device=self.scale_on_device,
        )


class FitsioDataset(Dataset):
    def __init__(self, filepaths: List[str], *, ext: int | None = None):
        self.filepaths = filepaths
        self.ext = ext

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        kwargs = {"ext": self.ext} if self.ext is not None else {}
        data = fitsio.read(self.filepaths[idx], **kwargs)
        return torch.from_numpy(data)


def warm_page_cache(filepaths: List[str], chunk_bytes: int = 8 * 1024 * 1024) -> None:
    """
    Prime OS page cache by sequentially reading each file.

    This is intentionally applied symmetrically before each backend run.
    """
    for path in filepaths:
        with open(path, "rb", buffering=0) as f:
            while f.read(chunk_bytes):
                pass


def benchmark_loader(
    dataset_factory: Callable[[], Dataset],
    *,
    batch_size: int = 4,
    num_workers: int = 2,
    device: str = "cpu",
    n_epochs: int = 3,
    description: str = "",
) -> float:
    dataset = dataset_factory()
    pin_memory = device == "cuda"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # Warm up workers/collation.
    for _ in loader:
        break

    start_time = time.time()
    total_pixels = 0
    for _ in range(n_epochs):
        for batch in loader:
            if device == "cuda":
                batch = batch.to(device, non_blocking=True)
            total_pixels += batch.numel()

    total_time = time.time() - start_time
    pixels_sec = total_pixels / total_time
    mb_sec = (total_pixels * 4) / (1024 * 1024) / total_time  # float32-equivalent
    print(f"{description:<28}: {pixels_sec:>12.0f} pix/sec ({mb_sec:>7.1f} mb/s)")
    return pixels_sec


def summarize_method(name: str, values: List[float]) -> None:
    med = median(values)
    avg = mean(values)
    vmin = min(values)
    vmax = max(values)
    print(
        f"{name:<28}: median={med:>12.0f}  mean={avg:>12.0f}  range=[{vmin:>12.0f}, {vmax:>12.0f}] pix/sec"
    )


def run_comparison(
    *,
    title: str,
    methods: Dict[str, Callable[[], Dataset]],
    filepaths: List[str],
    repeats: int,
    warm_cache: bool,
    warm_chunk_mb: int,
    batch_size: int,
    num_workers: int,
    device: str,
    n_epochs: int,
) -> Dict[str, List[float]]:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    names = list(methods.keys())
    out: Dict[str, List[float]] = {name: [] for name in names}

    for run_idx in range(repeats):
        order = names if (run_idx % 2 == 0) else list(reversed(names))
        print(f"\nrun {run_idx + 1}/{repeats} order: {' -> '.join(order)}")
        for name in order:
            if warm_cache:
                warm_page_cache(filepaths, chunk_bytes=warm_chunk_mb * 1024 * 1024)
            throughput = benchmark_loader(
                methods[name],
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                n_epochs=n_epochs,
                description=name,
            )
            out[name].append(throughput)

    print("\nsummary")
    for name in names:
        summarize_method(name, out[name])

    if "torchfits" in out and "fitsio + numpy" in out:
        speedup = median(out["torchfits"]) / median(out["fitsio + numpy"])
        print(f"median speedup (torchfits/fitsio + numpy): {speedup:.3f}x")
    if "torchfits (comp)" in out and "fitsio (comp)" in out:
        speedup = median(out["torchfits (comp)"]) / median(out["fitsio (comp)"])
        print(f"median speedup (torchfits (comp)/fitsio (comp)): {speedup:.3f}x")

    return out


def parse_shape(value: str) -> Tuple[int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise ValueError("--shape must be 'H,W', e.g. 2048,2048")
    h, w = int(parts[0]), int(parts[1])
    if h <= 0 or w <= 0:
        raise ValueError("shape dimensions must be positive")
    return (h, w)


def run_benchmark() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--n-files", type=int, default=50)
    parser.add_argument("--shape", type=str, default="2048,2048")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warm-chunk-mb", type=int, default=8)
    parser.add_argument(
        "--warm-cache",
        dest="warm_cache",
        action="store_true",
        help="Prime OS page cache before each backend run (default: on)",
    )
    parser.add_argument("--no-warm-cache", dest="warm_cache", action="store_false")
    parser.set_defaults(warm_cache=True)
    args = parser.parse_args()

    shape = parse_shape(args.shape)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda not available, falling back to cpu")
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("mps not available, falling back to cpu")
        device = "cpu"

    print(f"running benchmark on device: {device}")
    print(
        f"config: files={args.n_files} shape={shape} batch={args.batch_size} workers={args.num_workers} "
        f"epochs={args.epochs} repeats={args.repeats} warm_cache={args.warm_cache}"
    )

    tmp_dir = ""
    try:
        tmp_dir, filepaths = create_synthetic_dataset(args.n_files, shape)
        run_comparison(
            title="benchmark 1: uncompressed fits",
            methods={
                "torchfits": lambda: TorchFitsDataset(
                    filepaths,
                    hdu="auto",
                    mmap="auto",
                    cache_capacity=0,
                    handle_cache_capacity=64,
                    scale_on_device=True,
                ),
                "fitsio + numpy": lambda: FitsioDataset(filepaths, ext=None),
            },
            filepaths=filepaths,
            repeats=args.repeats,
            warm_cache=args.warm_cache,
            warm_chunk_mb=args.warm_chunk_mb,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            n_epochs=args.epochs,
        )
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    tmp_dir_comp = ""
    try:
        tmp_dir_comp, filepaths_comp = create_compressed_dataset(args.n_files, shape)
        run_comparison(
            title="benchmark 2: compressed fits (rice)",
            methods={
                # For compressed images, keep auto HDU detection to model user-facing defaults.
                "torchfits (comp)": lambda: TorchFitsDataset(
                    filepaths_comp,
                    hdu="auto",
                    mmap="auto",
                    cache_capacity=0,
                    handle_cache_capacity=64,
                    scale_on_device=True,
                ),
                # Explicit ext=1 is the payload HDU for these generated compressed files.
                "fitsio (comp)": lambda: FitsioDataset(filepaths_comp, ext=1),
            },
            filepaths=filepaths_comp,
            repeats=args.repeats,
            warm_cache=args.warm_cache,
            warm_chunk_mb=args.warm_chunk_mb,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            n_epochs=args.epochs,
        )
    finally:
        if tmp_dir_comp and os.path.exists(tmp_dir_comp):
            shutil.rmtree(tmp_dir_comp)


if __name__ == "__main__":
    run_benchmark()
