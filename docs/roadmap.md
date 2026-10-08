# Roadmap

torchfits reads and writes FITS as tensors and tables. World coordinates, units, and source finding stay in other packages.

## Where things stand

**1.2.0rc1** (2026-10-07) is the current release, on PyTorch 2.13. It is a candidate, not the final 1.2.0. The [changelog](changelog.md) has the notes. The changes that matter for what you call:

- Header and metadata reads go through `libtorchfits_core` and do not load PyTorch.
- Arrow reads of fixed columns and flat variable-length arrays do not import PyTorch. Tensor reads stay on `read_torch`.
- Transforms can carry inverse-variance and masks, and they refuse to calibrate data that is already calibrated.
- Scaled BITPIX=32 and 64 images are read as float64. Scaled BITPIX=8 and 16, including `quantize="robust"`, are float32. The 1.2.0rc1 wheels still return float64 for those 8- and 16-bit reads.

**1.1** added checksummed writes, streaming `where=` filters, and the remote downloader. **1.0** (2026-08-09) was memory-mapped image reads, table filters, the datasets, and the CLI.

## Next

- Buffered table reads (`mmap=False`) are still behind fitsio: narrow `read_full` by 22% on CPU and 13% on CUDA, and `ascii_10000` `predicate_filter` by 15% on CPU. Figures are the October 2026 runs in [Benchmarks](benchmarks.md#performance-deficits).
- Arrow conversion still copies native buffers through NumPy. PyArrow itself imports NumPy; torchfits should stop adding its own copy.
- `compress` and `decompress` only re-encode bytes, but they still load the torch-linked library.
- `torchfits.open()` still opens that library. Telling a table HDU from an image HDU also imports PyTorch, so a header-only open is not torch-free yet.

## Later

A later major version may read tiles with its own codecs and move bytes from storage into GPU memory without a host copy. That is not scheduled. The 1.x read and write calls stay as they are.

## Out of scope

- Celestial coordinates and WCS (`astropy.wcs`)
- Physical units (`astropy.units`)
- Source extraction, PSF fitting, and continuum fitting
- Anything outside the FITS standard
