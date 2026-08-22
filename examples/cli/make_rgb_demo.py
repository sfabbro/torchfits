#!/usr/bin/env python
"""Build a vivid 3-band RGB PNG via ``torchfits convert`` (synthetic demo)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torchfits  # noqa: E402
from torchfits.transforms.rgb import write_rgb_image  # noqa: E402
from torchfits.transforms import lupton_rgb  # noqa: E402


def _band(h: int, w: int, cx: float, cy: float, amp: float) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return amp * torch.exp(-r2 / 0.08) + 0.02 * torch.rand(h, w)


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/assets/gallery")
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = 256, 256
    r = _band(h, w, -0.25, -0.1, 1.2)
    g = _band(h, w, 0.05, 0.2, 1.0)
    b = _band(h, w, 0.3, -0.15, 1.1)
    tmp = out_dir / "_rgb_demo"
    tmp.mkdir(exist_ok=True)
    r_path, g_path, b_path = tmp / "r.fits", tmp / "g.fits", tmp / "b.fits"
    torchfits.write(str(r_path), r.to(torch.float32), overwrite=True)
    torchfits.write(str(g_path), g.to(torch.float32), overwrite=True)
    torchfits.write(str(b_path), b.to(torch.float32), overwrite=True)
    rgb = lupton_rgb(r, g, b, Q=6.0, stretch=0.4, minimum=0.0)
    png = out_dir / "cli_rgb_demo.png"
    write_rgb_image(str(png), rgb)
    print("wrote", png)
    print(
        "CLI equivalent:\n"
        f"  torchfits convert {r_path} {g_path} {b_path} {png} "
        "--to png --recipe lupton --q 6 --stretch 0.4"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
