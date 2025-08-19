# torchfits

High-performance FITS I/O for PyTorch with native pytorch-frame integration.

## Overview

torchfits is a hybrid C++/Python library designed for maximum performance in astronomical data loading pipelines. It provides:

- **Zero-copy tensor operations** with direct cfitsio integration
- **Batch coordinate transformations** using wcslib with OpenMP parallelization  
- **Native pytorch-frame support** for tabular data with lazy query building
- **Multi-level caching** (L1 memory + L2 disk) for remote files
- **Streaming datasets** for large-scale machine learning workflows

## Key Features

### 🚀 Performance-First Design
- C++ engine bypasses Python GIL for I/O operations
- SIMD-optimized data type conversions
- Tile-aware reading for compressed images
- Memory pools for efficient allocation

### 🔗 Deep PyTorch Integration
- Direct tensor creation from FITS data
- Seamless pytorch-frame TensorFrame support
- GPU-ready data loading with device placement
- PyTorch DataLoader compatibility

### 🌐 Remote Data Support
- Native HTTP/HTTPS/FTP protocol support via cfitsio
- Intelligent caching for remote files
- Streaming datasets for cloud-scale data

## Installation

### Using pixi (Recommended)

```bash
# Clone the repository
git clone https://github.com/sfabbro/torchfits.git
cd torchfits

# Set up development environment
pixi install
pixi run dev
```

### Using pip

```bash
pip install torchfits
```

## Quick Start

### Reading FITS Files

```python
import torchfits

# Smart function - returns Tensor for images, TensorFrame for tables
data = torchfits.read("image.fits", device='cuda')

# Advanced multi-HDU operations
with torchfits.open("multi_hdu.fits") as hdul:
    image = hdul[0].to_tensor(device='cuda')
    table = hdul[1].materialize()  # Returns TensorFrame
```

### Working with Images

```python
# Lazy data access with slicing
with torchfits.open("large_image.fits") as hdul:
    hdu = hdul[0]
    
    # Get subset without loading full image
    cutout = hdu.data[1000:2000, 1000:2000]  # Returns Tensor
    
    # Memory-efficient statistics
    stats = hdu.stats()  # Computed in C++
    
    # WCS transformations
    pixels = torch.tensor([[100, 200], [300, 400]])
    world_coords = hdu.wcs.pixel_to_world(pixels)
```

### Working with Tables

```python
# Lazy, chainable query building (torch-frame native)
with torchfits.open("catalog.fits") as hdul:
    table = hdul[1]
    
    # Build query plan (no I/O yet)
    query = (table
             .select(['RA', 'DEC', 'MAG_G'])
             .filter("MAG_G < 20 AND FLAG == 0")
             .head(10000))
    
    # Execute and get TensorFrame
    df = query.materialize()
    
    # Or stream in batches
    for batch in query.iter_rows(batch_size=1000):
        process_batch(batch)
```

### Machine Learning Workflows

```python
from torchfits import create_dataloader

# Create optimized DataLoader
file_paths = ["image1.fits", "image2.fits", ...]
dataloader = create_dataloader(
    file_paths, 
    batch_size=32,
    num_workers=4,
    device='cuda'
)

# Training loop
for batch in dataloader:
    # batch is already on GPU
    loss = model(batch)
    loss.backward()
```

## API Reference

### Top-Level Functions

- `torchfits.read(path, hdu=0, device='cpu')` - Smart read function
- `torchfits.write(path, data, header=None)` - Smart write function  
- `torchfits.open(path, mode='r')` - Multi-HDU file access

### Core Classes

- `HDUList` - Container for multiple HDUs with context management
- `TensorHDU` - Image/cube data with lazy loading and WCS support
- `TableHDU` - Tabular data with torch-frame integration
- `WCS` - Batch coordinate transformations

### Machine Learning

- `FITSDataset` - Map-style dataset for random access
- `IterableFITSDataset` - Streaming dataset for large-scale data
- `create_dataloader()` - Factory for optimized DataLoaders

## Performance

torchfits is designed for maximum performance:

- **10-100x faster** than astropy for large arrays
- **Zero-copy** tensor creation from FITS data
- **Parallel I/O** with OpenMP acceleration
- **Optimized memory usage** with shared buffers

## Development

### Building from Source

```bash
# Set up development environment
pixi install

# Build C++ extension
pixi run build

# Run tests
pixi run test

# Run benchmarks
pixi run bench
```

### Project Structure

```
src/torchfits/
├── __init__.py          # Top-level API
├── hdu.py              # Core HDU classes
├── wcs.py              # WCS functionality
├── datasets.py         # PyTorch datasets
├── dataloader.py       # DataLoader factories
└── cpp/                # C++ extension
    ├── fits.cpp        # FITS I/O engine
    ├── wcs.cpp         # WCS transformations
    └── bindings.cpp    # pybind11 interface
```

## License

GPL-2 License. See LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.