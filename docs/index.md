---
template: home.html
title: FITS I/O for tensors and tables
---

<div class="tf-below" markdown>

## Browse

<ul class="tf-paths" markdown>
<li markdown>
[Install](install.md)

One-line installs: default CUDA+CPU, CPU-only, or pinned CUDA builds.
</li>
<li markdown>
[Quick start](quickstart.md)

Images, tables, cutouts, and the CLI.
</li>
<li markdown>
[Python workflows](python-workflows.md)

Which API for images, catalogs, cutouts, and datasets.
</li>
<li markdown>
[CLI](cli.md)

`info`, `header`, `verify`, `cutout`, …
</li>
<li markdown>
[API](api.md)

Core I/O, tables, datasets, transforms.
</li>
<li markdown>
[Examples](examples.md)

Runnable scripts and transform galleries.
</li>
<li markdown>
[Benchmarks](benchmarks.md)

Methodology and scorecards.
</li>
<li markdown>
[Migration](migration_astropy.md) · [fitsio](migration_fitsio.md)

Side-by-side guides from Astropy and fitsio.
</li>
<li markdown>
[Parity](parity.md)

What torchfits covers today.
</li>
</ul>

## At a glance

```python
import torchfits

tensor = torchfits.read_tensor("image.fits", hdu=0)
table = torchfits.table.read("catalog.fits", hdu=1, where="MAG_G < 20")
cutout = torchfits.read_subset("image.fits", hdu=0, x1=100, y1=100, x2=200, y2=200)
```

```bash
torchfits info image.fits
torchfits header image.fits
```

## Why torchfits?

torchfits cuts **1.2.3** next (not yet released; **1.0.0rc5** is on PyPI) — see [Changelog](changelog.md) and
[Benchmarks](benchmarks.md#performance-deficits) for scope and known lags.

| | astropy / fitsio | torchfits |
|---|---|---|
| **Device placement** | manual `.to(device)` | `device="cuda"` / `"mps"` / `"cpu"` |
| **Table filtering** | Python mask | `where=` on `table.read` |
| **Shell tooling** | fitsinfo / fitsheader / … | `torchfits` CLI |

Current performance results and methodology live in [Benchmarks](benchmarks.md);
the landing page intentionally avoids copying volatile benchmark numbers.
Datasets and loaders for multi-file work live under `torchfits.data` — see
[Examples → ML with FITS](examples-ml.md) when you need them.

Docs channels: [stable](https://astroai.github.io/torchfits/) (latest `v*` tag) ·
[edge](https://astroai.github.io/torchfits/edge/) (`main` tip).

</div>
