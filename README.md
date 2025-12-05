# torchfits

[![PyPI version](https://badge.fury.io/py/torchfits.svg)](https://badge.fury.io/py/torchfits)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

High-performance FITS I/O for PyTorch with zero-copy tensor operations and native GPU support.

## Features

- **🚀 Fast**: 10-100x faster than astropy for large arrays with zero-copy tensor creation
- **🎯 PyTorch Native**: Direct tensor creation on CPU, CUDA, or MPS devices
- **📊 Table Support**: Read FITS tables as dictionaries of tensors with column selection
- **🌍 WCS Integration**: Batch coordinate transformations with wcslib
- **🔄 Transforms**: GPU-accelerated astronomical data transformations
- **💾 Smart Caching**: Multi-level caching for remote files and repeated access

## Installation

### From PyPI

```bash
pip install torchfits
```

### From Source

```bash
git clone https://github.com/sfabbro/torchfits.git
cd torchfits
pip install -e .
```

### Development Setup

```bash
# Using pixi (recommended)
pixi install
pixi run dev
```

## Quick Start

### Reading Images

```python
import torchfits

# Read FITS image as PyTorch tensor
data, header = torchfits.read("image.fits", device='cuda')
print(data.shape, data.device)  # torch.Size([2048, 2048]) cuda:0

# Read specific HDU
data, header = torchfits.read("multi.fits", hdu=1)

# Read subset (cutout)
cutout = torchfits.read_subset("large.fits", hdu=0, 
                               x1=100, y1=100, x2=200, y2=200)
```

### Reading Tables

```python
# Read FITS table as dictionary of tensors
table, header = torchfits.read("catalog.fits", hdu=1)

# Access columns
ra = table['RA']      # torch.Tensor
dec = table['DEC']    # torch.Tensor
mag = table['MAG_G']  # torch.Tensor

# Select specific columns
table, _ = torchfits.read("catalog.fits", hdu=1, 
                          columns=['RA', 'DEC', 'MAG_G'])

# Read row range
table, _ = torchfits.read("catalog.fits", hdu=1,
                          start_row=1000, num_rows=5000)
```

### Writing FITS Files

```python
import torch

# Write tensor as FITS image
data = torch.randn(512, 512)
torchfits.write("output.fits", data, overwrite=True)

# Write with header
header = {'OBJECT': 'M31', 'EXPTIME': 300.0}
torchfits.write("output.fits", data, header=header, overwrite=True)

# Write table
table = {
    'RA': torch.randn(1000),
    'DEC': torch.randn(1000),
    'MAG': torch.randn(1000)
}
torchfits.write("catalog.fits", table, overwrite=True)
```

### Data Transformations

```python
from torchfits.transforms import ZScale, AsinhStretch, Compose

# Create transformation pipeline
transform = Compose([
    ZScale(),           # Normalize to [0, 1]
    AsinhStretch(),     # Asinh stretch for display
])

# Apply to data (works on GPU!)
data, _ = torchfits.read("image.fits", device='cuda')
stretched = transform(data)
```

### Machine Learning Workflows

```python
from torchfits import FITSDataset, create_dataloader

# Create dataset
dataset = FITSDataset(
    file_paths=["img1.fits", "img2.fits", ...],
    transform=transform,
    device='cuda'
)

# Create DataLoader
dataloader = create_dataloader(
    dataset,
    batch_size=32,
    num_workers=4
)

# Training loop
for batch in dataloader:
    # batch is already on GPU
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
```

## Performance

torchfits is designed for maximum performance:

| Operation | torchfits | astropy | Speedup |
|-----------|-----------|---------|---------|
| Read 2k×2k image | 3ms | 45ms | **15x** |
| Read 1M row table | 12ms | 850ms | **70x** |
| WCS transform (1M points) | 8ms | 420ms | **52x** |

*Benchmarks on M2 MacBook Air. See `benchmarks/` for details.*

## Documentation

- **[API Reference](API.md)** - Complete API documentation with examples
- **[Examples](examples/)** - Working examples for common use cases
- **[CHANGELOG](CHANGELOG.md)** - Version history and changes

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.0
  
## Device Support

torchfits supports multiple compute devices:

- **CPU**: Standard CPU tensors
- **CUDA**: NVIDIA GPU acceleration
- **MPS**: Apple Silicon GPU acceleration (M1/M2/M3)

```python
# Specify device when reading
data, _ = torchfits.read("image.fits", device='mps')  # Apple Silicon
data, _ = torchfits.read("image.fits", device='cuda') # NVIDIA GPU
data, _ = torchfits.read("image.fits", device='cpu')  # CPU
```

## License

GPL-2.0 License. See [LICENSE](LICENSE) for details.

## Citation

If you use torchfits in your research, please cite:

```bibtex
@software{torchfits2025,
  author = {Fabbro, Seb},
  title = {torchfits: High-performance FITS I/O for PyTorch},
  year = {2025},
  url = {https://github.com/sfabbro/torchfits}
}
```

## Acknowledgments

Built with:
- [cfitsio](https://heasarc.gsfc.nasa.gov/fitsio/) - FITS file I/O library
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [nanobind](https://github.com/wjakob/nanobind) - C++/Python bindings
