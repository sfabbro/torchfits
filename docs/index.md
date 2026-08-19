---
template: home.html
title: FITS I/O for PyTorch Tensors and Tables
---

<div class="tf-below" markdown>

**`torchfits`** delivers high-performance FITS I/O for the modern Python data science and machine learning ecosystem. Powered by a native C++ engine with vendored CFITSIO, it decodes astronomical images directly to PyTorch GPU/CPU tensors, queries binary catalogs as zero-copy PyArrow tables, extracts survey cutouts in sub-milliseconds, and provides a full shell CLI.

Prebuilt binary wheels are available for **Linux (x86_64, aarch64)** and **macOS (Apple Silicon arm64)** supporting **Python 3.10–3.14** (no C++ compiler or system libraries needed).

---

## Core Capabilities

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **C++ Engine & Vendored CFITSIO**

    ---

    Decodes FITS structures at near-native C speed. Ships with CFITSIO vendored in prebuilt wheels — install and run with zero compiler setup.

    [:octicons-arrow-right-24: Installation Guide](install.md)

-   :material-memory:{ .lg .middle } **Direct GPU Tensor Placement**

    ---

    Load 2D and 3D images directly onto CUDA GPUs or Apple Silicon unified memory (`device="cuda"` / `"mps"`) in a single call.

    [:octicons-arrow-right-24: Core I/O Workflows](python-workflows.md#images-and-hdus)

-   :material-database-search:{ .lg .middle } **Arrow-Native Catalogs & SQL**

    ---

    Filter multi-million row catalogs with C++ pushdown predicates (`where="MAG < 20"`). Zero-copy integration with Polars, Pandas, and DuckDB.

    [:octicons-arrow-right-24: Table Workflows](python-workflows.md#tables-as-dataframes)

-   :material-telescope:{ .lg .middle } **Survey-Scale Cutouts & ML**

    ---

    Extract sub-regions and postage stamps from giant mosaics without RAM exhaustion. Native PyTorch `Dataset` and multi-worker `make_loader`.

    [:octicons-arrow-right-24: Machine Learning with FITS](examples-ml.md)

</div>

---

## At a Glance

=== "Images to Tensors"

    ```python
    import torchfits

    # Read image directly onto GPU or CPU
    tensor = torchfits.read_tensor("science.fits", hdu="SCI", device="cuda")
    print(f"Loaded tensor on {tensor.device} with shape {tensor.shape}")

    # Read pixel data alongside header cards
    data, header = torchfits.read("science.fits", return_header=True)
    print("Object:", header.get("OBJECT"))
    ```

=== "Catalog Filtering & DataFrames"

    ```python
    import torchfits

    # Fast C++ predicate pushdown into PyArrow Table
    table = torchfits.table.read(
        "catalog.fits",
        hdu=1,
        columns=["RA", "DEC", "MAG_G"],
        where="MAG_G < 20.0 AND CLASS_STAR > 0.8",
    )

    # Convert to Polars or Pandas without data copies
    import polars as pl

    df = pl.from_arrow(table)
    ```

=== "Rapid Survey Cutouts"

    ```python
    import torchfits

    # Single bounding box cutout [x1, y1, x2, y2)
    stamp = torchfits.read_subset("mosaic.fits", hdu=0, x1=100, y1=100, x2=228, y2=228)

    # High-throughput batch cutouts using reusable file handle
    with torchfits.open_subset_reader("survey_mosaic.fits", hdu=0) as reader:
        stamp_a = reader.read_subset(100, 100, 228, 228)
        stamp_b = reader.read_subset(500, 500, 628, 628)
    ```

=== "Shell CLI Tool"

    ```bash
    # Inspect extensions, shapes, and data types
    torchfits info science.fits

    # Dump header keywords or build cross-file summary catalogs
    torchfits header *.fits --keyword-table -k OBJECT -k FILTER

    # Filter binary tables and export directly to Apache Parquet
    torchfits convert catalog.fits bright.parquet -e 1 -w "MAG_G < 18.0"
    ```

---

## Why torchfits?

| Capability | Astropy (`io.fits`) / `fitsio` | `torchfits` |
|---|---|---|
| **GPU Tensor Placement** | Manual host read $\rightarrow$ `.to(device)` copy | Native `device="cuda"` / `"mps"` decode |
| **Catalog Query & Slicing** | Load full table to Python $\rightarrow$ boolean mask | C++ in-engine `where=` pushdown to PyArrow |
| **Mosaic Cutouts** | Section indexing or full-frame slicing | Zero-overhead `read_subset` & `open_subset_reader` |
| **Machine Learning Pipelines** | Custom boilerplate wrapper classes | Native `FitsImageDataset` + multi-worker `make_loader` |
| **Command-Line Suite** | Disparate tools (`fitsinfo`, `imstat`, `fpack`) | Unified, multi-core `torchfits` CLI suite |
| **Packaging & Installation** | May require local C compilation | Prebuilt wheels with vendored CFITSIO |

Explore detailed benchmarks and scorecards in [Benchmarks](benchmarks.md), or check feature coverage in the [Compatibility & Parity Matrix](parity.md).

---

## Documentation Pathways

<div class="grid cards" markdown>

-   **Getting Started**

    ---

    - [Installation Guide](install.md) — Prebuilt wheels, CPU-only, CUDA builds, and source.
    - [Quick Start](quickstart.md) — 5-minute tour of core operations.
    - [Compatibility Matrix](compatibility.md) — Python, PyTorch, and platform support.

-   **User Guides**

    ---

    - [Python Workflows](python-workflows.md) — Best practices for images, tables, and datasets.
    - [CLI Guide](cli.md) — Complete 15-command shell reference.
    - [CLI Recipes](cli-recipes.md) — Practical, copy-pasteable shell workflows.
    - [Transforms Gallery](examples-transforms.md) — Astronomical stretch and normalization gallery.

-   **Reference & Internals**

    ---

    - [API Reference](api.md) — Full signatures, parameters, and return types.
    - [Core I/O](api-core-io.md) & [Tables API](api-tables.md) — Detailed class & function docs.
    - [Benchmarks](benchmarks.md) — Performance methodology and hardware scorecards.
    - [Astropy](migration_astropy.md) · [fitsio Migration](migration_fitsio.md) — Side-by-side transition guides.

</div>

<br>

<small>Docs channels: [stable](https://astroai.github.io/torchfits/) (latest release) · [edge](https://astroai.github.io/torchfits/edge/) (`main` tip).</small>

</div>
