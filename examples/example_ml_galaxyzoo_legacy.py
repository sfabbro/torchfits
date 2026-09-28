"""Example: Galaxy Zoo 1 labels + Legacy Survey grz cutouts -> tiny CNN.

Real labels (Galaxy Zoo 1 DR table2 FITS) + real multi-band FITS cutouts
(Legacy Survey ``fits-cutout`` service) — no HDF5 / Galaxy10 bundle.

Filters ``UNCERTAIN == 0`` and ``SPIRAL == 1 or ELLIPTICAL == 1``, downloads
up to ``GZ_N`` (default 200) grz cutouts keyed by RA/Dec, trains one epoch of
a tiny CNN, and prints loss/accuracy. Skips cleanly under
``TORCHFITS_EXAMPLE_FAST=1`` or when the network/samples are unavailable.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from examples._sample_data import (  # noqa: E402
    SampleUnavailable,
    gz_legacy_cutouts_dir,
    try_ensure_sample,
)

import torchfits  # noqa: E402
from torchfits import table as tf_table  # noqa: E402
from torchfits.transforms.rgb import write_rgb_image  # noqa: E402
from torchfits.data import FitsImageDataset, make_loader  # noqa: E402
from torchfits.transforms import (  # noqa: E402
    ArcsinhStretch,
    BackgroundSubtract,
    Compose,
    FITSTransform,
    ZScaleNormalize,
    lupton_rgb,
)

SIZE = 64
DEFAULT_GZ_N = 200
# Per-cutout wall-clock cap. urlretrieve() with no timeout blocks indefinitely on
# a stalled connection, and the examples runner's only backstop is its 300s
# per-example kill, which fails the whole example. A socket timeout instead
# raises TimeoutError -- already an OSError, so the handler below skips just this
# cutout and the example degrades to the cutouts that did arrive.
CUTOUT_TIMEOUT_S = 20.0
CUTOUT_URL = (
    "https://www.legacysurvey.org/viewer/fits-cutout"
    "?ra={ra}&dec={dec}&layer=ls-dr10-south&pixscale=0.262&bands=grz&size={size}"
)


def _fast_mode() -> bool:
    return os.environ.get("TORCHFITS_EXAMPLE_FAST", "").strip() in (
        "1",
        "true",
        "TRUE",
        "yes",
    )


class NanToZero(FITSTransform):
    """Replace NaN pixels (Legacy Survey off-footprint gaps) with 0.

    NOTE: not meaningfully invertible; ``inverse`` is identity.
    """

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return x


def _hms_to_deg(s: str) -> float:
    h, m, sec = (float(v) for v in s.split(":"))
    return (h + m / 60.0 + sec / 3600.0) * 15.0


def _dms_to_deg(s: str) -> float:
    s = s.strip()
    sign = -1.0 if s.startswith("-") else 1.0
    d, m, sec = (float(v) for v in s.lstrip("+-").split(":"))
    return sign * (d + m / 60.0 + sec / 3600.0)


def _ensure_cutout(
    idx: int, ra_deg: float, dec_deg: float, dest_dir: Path
) -> Path | None:
    dest = dest_dir / f"{idx:04d}.fits"
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    url = CUTOUT_URL.format(ra=ra_deg, dec=dec_deg, size=SIZE)
    try:
        # urlopen, not urlretrieve: urlretrieve takes no timeout and blocks
        # indefinitely on a stalled connection.
        with urllib.request.urlopen(url, timeout=CUTOUT_TIMEOUT_S) as response:  # noqa: S310 — fixed public API
            tmp.write_bytes(response.read())
        tmp.replace(dest)
        return dest
    except (urllib.error.URLError, OSError):
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return None


class TinyCNN(nn.Module):
    def __init__(self, size: int = SIZE) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * (size // 4) * (size // 4), 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def _save_class_grid(paths: list[Path], labels: list[int]) -> None:
    """Write a 4x4 Lupton RGB grid (raw grz), not zscale wash."""
    n = min(16, len(paths))
    tiles = []
    for path in paths[:n]:
        img = torchfits.read_tensor(str(path), hdu=0).float()
        img = torch.nan_to_num(img, nan=0.0)
        # Legacy Survey fits-cutout bands=grz → [g,r,z]; Lupton wants R,G,B = z,r,g
        g, r, z = img[0], img[1], img[2]
        tiles.append(lupton_rgb(r=z, g=r, b=g, Q=8.0, stretch=0.3).float())
    cols = 4
    rows = (n + cols - 1) // cols
    canvas = torch.zeros((rows * SIZE, cols * SIZE, 3), dtype=torch.float32)
    for i, tile in enumerate(tiles):
        rr, cc = divmod(i, cols)
        canvas[rr * SIZE : (rr + 1) * SIZE, cc * SIZE : (cc + 1) * SIZE, :] = tile
    out_path = Path(__file__).resolve().parent / "output" / "ml_gz_class_grid.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_rgb_image(str(out_path), canvas)
    print(f"wrote {out_path} (labels: {labels[:n]})")


def main() -> int:
    if _fast_mode():
        print("SKIP: TORCHFITS_EXAMPLE_FAST=1")
        return 0

    try:
        table_path = try_ensure_sample("galaxy_zoo1_table2")
    except SampleUnavailable as exc:
        print(f"SKIP: {exc}")
        return 0
    if table_path is None:
        print("SKIP: galaxy_zoo1_table2 sample unavailable (no network / FAST mode)")
        return 0

    table = tf_table.read(str(table_path), hdu=1)
    mask = pc.and_(
        pc.equal(table["UNCERTAIN"], 0),
        pc.or_(pc.equal(table["SPIRAL"], 1), pc.equal(table["ELLIPTICAL"], 1)),
    )
    filtered = table.filter(mask)
    print(f"catalog: {table.num_rows} rows -> {filtered.num_rows} labeled, unambiguous")
    if filtered.num_rows == 0:
        print("SKIP: no rows passed the SPIRAL/ELLIPTICAL/UNCERTAIN filter")
        return 0

    n_want = int(os.environ.get("GZ_N", DEFAULT_GZ_N))
    rng = np.random.default_rng(0)
    n_sample = min(n_want, filtered.num_rows)
    chosen = rng.choice(filtered.num_rows, size=n_sample, replace=False)
    sub = filtered.take(list(chosen)).to_pydict()

    dest_dir = gz_legacy_cutouts_dir()
    paths: list[Path] = []
    labels: list[int] = []
    for i in range(n_sample):
        ra_deg = _hms_to_deg(sub["RA"][i])
        dec_deg = _dms_to_deg(sub["DEC"][i])
        path = _ensure_cutout(i, ra_deg, dec_deg, dest_dir)
        if path is None:
            continue
        paths.append(path)
        labels.append(int(sub["SPIRAL"][i]))  # 1=spiral, 0=elliptical

    print(f"cutouts: {len(paths)}/{n_sample} downloaded to {dest_dir}")
    if not paths:
        print("SKIP: no Legacy Survey cutouts available (no network?)")
        return 0

    pipeline = Compose(
        [NanToZero(), BackgroundSubtract(), ArcsinhStretch(a=0.1), ZScaleNormalize()]
    )
    dataset = FitsImageDataset(
        [str(p) for p in paths], hdu=0, labels=labels, transform=pipeline, device="cpu"
    )
    n_spiral = sum(labels)
    print(
        f"dataset: n={len(dataset)} spiral={n_spiral} elliptical={len(labels) - n_spiral}"
    )

    loader = make_loader(dataset, batch_size=16, num_workers=0, optimize_cache=False)

    model = TinyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    total_loss, correct, n_seen = 0.0, 0, 0
    for images, batch_labels in loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == batch_labels).sum().item()
        n_seen += images.size(0)
    print(
        f"epoch 1: loss={total_loss / n_seen:.4f} acc={correct / n_seen:.3f} (n={n_seen})"
    )

    try:
        _save_class_grid(paths, labels)
    except Exception as exc:  # pragma: no cover - best-effort visualization
        print(f"note: class grid PNG skipped ({exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
