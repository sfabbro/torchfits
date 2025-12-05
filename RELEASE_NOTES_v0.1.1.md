# torchfits v0.1.1

First release on PyPI.

## Installation

```bash
pip install torchfits
```

**Requirements:**
- Python ≥ 3.11
- PyTorch ≥ 2.0.0
- System: cfitsio, wcslib

## What's New in v0.1.1

### Documentation
- Added PyTorch-Frame integration documentation to README.md and API.md
- New example: `examples/example_frame.py` demonstrating FITS table to TensorFrame conversion
- API documentation for `read_tensor_frame()`, `to_tensor_frame()`, `write_tensor_frame()`

## Features

### Core I/O
- Read FITS files to PyTorch tensors
- Write tensors to FITS files
- Direct GPU loading (CUDA, MPS)
- Memory-mapped reading for large files
- Multi-extension FITS support
- Compressed FITS reading (Rice, gzip)

### Table Operations
- Read FITS tables as dictionaries of tensors
- Column selection and row range reading
- PyTorch-Frame integration for tabular ML

### WCS and Transforms
- WCS coordinate transformations (pixel ↔ world)
- Astronomy transforms: ZScale, AsinhStretch, LogStretch, PowerStretch
- PyTorch Dataset/DataLoader integration

## Performance

Benchmarked against astropy and fitsio:
- 4k×4k image read: 1.3× faster than fitsio, 7.4× faster than astropy
- 1M row table: 1.7× faster than fitsio, 3.8× faster than astropy
- GPU transfer: 52× faster than CPU-based alternatives

## Quick Start

```python
import torchfits

# Read FITS image
data, header = torchfits.read("image.fits")

# Read to GPU
data_gpu, _ = torchfits.read("image.fits", device='cuda')

# Read table
table, _ = torchfits.read("catalog.fits", hdu=1)

# PyTorch-Frame integration
tf = torchfits.read_tensor_frame("catalog.fits", hdu=1)
```

## Documentation

- Repository: https://github.com/sfabbro/torchfits
- API Reference: https://github.com/sfabbro/torchfits/blob/main/API.md
- Examples: https://github.com/sfabbro/torchfits/tree/main/examples

## License

GNU General Public License v2.0 (GPL-2.0)
