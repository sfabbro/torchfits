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
[Migration](migration_astropy.md)

From Astropy / fitsio.
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

torchfits is a **1.0.0rc4** prerelease — see [Changelog](changelog.md) and
[Benchmarks](benchmarks.md#performance-deficits) for scope and known lags.

| | astropy / fitsio | torchfits |
|---|---|---|
| **Image read (16 MB, MPS)** | 8.16 ms / 3.67 ms | **3.85 ms** (~2× vs astropy; ~parity vs fitsio) |
| **Table read (100k rows, mixed)** | 31.85 ms / 10.44 ms | **2.20 ms** (~15× / ~5×) |
| **Repeated cutouts (50×)** | 86.01 ms / 5.37 ms | **0.79 ms** (~116× / ~7×) |
| **Device placement** | manual `.to(device)` | `device="cuda"` / `"mps"` / `"cpu"` |
| **Table filtering** | Python mask | `where=` on `table.read` |
| **Shell tooling** | fitsinfo / fitsheader / … | `torchfits` CLI |

Representative medians from Round-3 `exhaustive_mps_20260719_143706`
(methodology and deficits in [Benchmarks](benchmarks.md)). Datasets and
loaders for multi-file work live under `torchfits.data` — see
[Examples → ML with FITS](examples-ml.md) when you need them.

Docs channels: [stable](https://astroai.github.io/torchfits/) (latest `v*` tag) ·
[edge](https://astroai.github.io/torchfits/edge/) (`main` tip).

</div>
