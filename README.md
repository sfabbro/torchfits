# torchfits

[![PyPI version](https://badge.fury.io/py/torchfits.svg)](https://badge.fury.io/py/torchfits)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

High-performance FITS I/O library for PyTorch. Provides zero-copy tensor operations and native GPU support for astronomical data processing and machine learning workflows.

## Installation

```bash
pip install torchfits
```

## Features

- **Fast I/O**: Zero-copy tensor creation from FITS data with SIMD-optimized type conversions
- **Multi-device**: Direct tensor creation on CPU, CUDA, or MPS (Apple Silicon) devices
- **FITS tables**: Read binary tables as dictionaries of tensors with column and row selection
- **WCS support**: Batch coordinate transformations using wcslib with OpenMP parallelization (Optimized for 2D TAN/SIP/TPV)
- **Data transforms**: GPU-accelerated astronomical normalization and augmentation (ZScale, asinh stretch, etc.)
- **Smart caching**: Multi-level caching (L1 memory + L2 disk) for remote files and repeated access
- **FITS compliant**: Built on cfitsio for standards compliance and robust file handling

## Quick Start

### Reading Images

```python
import torchfits

# Read FITS image as PyTorch tensor
data, header = torchfits.read("image.fits", device='cuda')
print(data.shape, data.device)  # torch.Size([2048, 2048]) cuda:0

# Read specific HDU
data, header = torchfits.read("multi.fits", hdu=1)

# Read cutout from large file
cutout = torchfits.read_subset("large.fits", hdu=0, 
                               x1=100, y1=100, x2=200, y2=200)
```

### Reading Tables

```python
# Read FITS table as dictionary of tensors
table, header = torchfits.read("catalog.fits", hdu=1)

# Access columns as tensors
ra = table['RA']      # torch.Tensor
dec = table['DEC']    # torch.Tensor
mag = table['MAG_G']  # torch.Tensor

# Select specific columns and row ranges
table, _ = torchfits.read("catalog.fits", hdu=1, 
                          columns=['RA', 'DEC'],
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

### Data Processing

```python
from torchfits.transforms import ZScale, AsinhStretch, Compose

# Create transformation pipeline
transform = Compose([
    ZScale(),           # Normalize using IRAF ZScale algorithm
    AsinhStretch(),     # Asinh stretch for high dynamic range
])

# Apply transformations on GPU
data, _ = torchfits.read("image.fits", device='cuda')
stretched = transform(data)
```

### Machine Learning Integration

```python
from torchfits import FITSDataset, create_dataloader

# Create dataset
dataset = FITSDataset(
    file_paths=["img1.fits", "img2.fits", ...],
    transform=transform,
    device='cuda'
)

# Create DataLoader compatible with PyTorch training loops
dataloader = create_dataloader(dataset, batch_size=32, num_workers=4)

for batch in dataloader:
    # batch is already on GPU
    output = model(batch)
```

## Performance

Performance measurements on M2 MacBook Air with 2048×2048 float32 images and 1M row tables:

| Operation | torchfits | astropy | fitsio | Best |
|-----------|-----------|---------|--------|------|
| Read 2k×2k image | 3 ms | 45 ms | 5 ms | torchfits (1.7× vs fitsio) |
| Read 1M row table | 12 ms | 850 ms | 15 ms | torchfits (1.3× vs fitsio) |
| WCS transform (1M pts) | 8 ms | 420 ms | N/A | torchfits |

Performance characteristics:
- Zero-copy operations provide consistent speedup across data sizes
- SIMD-optimized type conversions reduce overhead (Optimized for float32)
- Direct GPU placement eliminates host-device transfer for ML workflows
- Competitive with fitsio while providing PyTorch tensor output
- Largest gains over astropy for tables and large arrays

See `benchmarks/` for detailed methodology and scaling behavior.

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.0
- cfitsio (bundled)
- wcslib (system dependency)

## Device Support

- **CPU**: Standard CPU tensors with SIMD acceleration
- **CUDA**: NVIDIA GPU acceleration  
- **MPS**: Apple Silicon GPU (M1/M2/M3) - note some overhead for small workloads

```python
# Specify device when reading
data, _ = torchfits.read("image.fits", device='mps')  # Apple Silicon
data, _ = torchfits.read("image.fits", device='cuda') # NVIDIA GPU
data, _ = torchfits.read("image.fits", device='cpu')  # CPU
```

For more examples, see the `examples/` directory.

## PyTorch-Frame Integration

Seamlessly convert FITS tables to [pytorch-frame](https://pytorch-frame.readthedocs.io/) `TensorFrame` objects for tabular deep learning:

```python
import torchfits

# Read FITS table directly as TensorFrame
tf = torchfits.read_tensor_frame("catalog.fits", hdu=1)

# Or convert from dict
data, header = torchfits.read("catalog.fits", hdu=1)
tf = torchfits.to_tensor_frame(data)

# Use with pytorch-frame models
print(tf.feat_dict)  # {stype.numerical: tensor, stype.categorical: tensor}
print(tf.col_names_dict)  # Column names grouped by semantic type

# Write TensorFrame back to FITS
torchfits.write_tensor_frame("output.fits", tf, overwrite=True)
```

Automatic semantic type inference:
- `float32/float64` → numerical features
- `int32/int16/uint8` → numerical features  
- `int64/bool` → categorical features

See `examples/example_frame.py` for a complete workflow.

## Documentation

- [API Reference](API.md) - Complete API documentation
- [Examples](examples/) - Working examples for common workflows
- [CHANGELOG](CHANGELOG.md) - Version history

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

- [cfitsio](https://heasarc.gsfc.nasa.gov/fitsio/) - FITS file I/O library
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [nanobind](https://github.com/wjakob/nanobind) - C++/Python bindings
