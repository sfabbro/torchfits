# Roadmap

torchfits reads and writes FITS as tensors and tables. World coordinates, units, and source finding stay in other packages.

## Where things stand

**1.2.3** (2026-10-20) is the current release, on the PyTorch 2.13 lane. The next minor, 1.3.0, moves to the PyTorch 2.14 lane and to Python 3.11–3.15 (3.10 is dropped). It also makes `torchfits.open()` and header/HDU inspection run without importing PyTorch, and wraps native table buffers in Arrow instead of staging them through NumPy. The [changelog](changelog.md) has the notes. The changes that matter for what you call:

- Header and metadata reads go through `libtorchfits_core` and do not load PyTorch.
- Arrow reads of fixed columns and flat variable-length arrays do not import PyTorch. Tensor reads stay on `read_torch`.
- Transforms can carry inverse-variance and masks, and they refuse to calibrate data that is already calibrated.
- Scaled BITPIX=32 and 64 images are read as float64. Scaled BITPIX=8 and 16, including `quantize="robust"`, are float32.
- The legacy `torchfits.cpp` module is removed; the native surface is internal (`torchfits._cpp`).

**1.1** added checksummed writes, streaming `where=` filters, and the remote downloader. **1.0** (2026-08-09) was memory-mapped image reads, table filters, the datasets, and the CLI.

## Next

- Buffered table reads (`mmap=False`) are still behind fitsio: narrow `read_full` by 22% on CPU and 13% on CUDA, and `ascii_10000` `predicate_filter` by 15% on CPU. Figures are the October 2026 runs published in [torchfits-harness](https://github.com/astroai/torchfits-harness).
- `compress` and `decompress` only re-encode bytes, but they still load the torch-linked library.
- Arrow table destinations avoid THPVariables, but the raw reader still lives in the torch-linked extension, so `table.read` imports PyTorch when it is installed; moving the raw reader into the core makes the destination truly torch-free.

## Later

A later major version may read tiles with its own codecs and move bytes from storage into GPU memory without a host copy. That is not scheduled. The 1.x read and write calls stay as they are.

## Out of scope

- Celestial coordinates and WCS (`astropy.wcs`)
- Physical units (`astropy.units`)
- Source extraction, PSF fitting, and continuum fitting
- Anything outside the FITS standard
