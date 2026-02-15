# torchfits

[![PyPI](https://img.shields.io/pypi/v/torchfits)](https://pypi.org/project/torchfits/)
[![Wheels](https://github.com/sfabbro/torchfits/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/sfabbro/torchfits/actions/workflows/build_wheels.yml)
[![CI](https://github.com/sfabbro/torchfits/actions/workflows/ci.yml/badge.svg)](https://github.com/sfabbro/torchfits/actions/workflows/ci.yml)

Fast FITS I/O for PyTorch image and table workflows.

## Features

- **Native PyTorch Integration**: Read FITS images and tables directly into `torch.Tensor` with zero-copy efficiency, supporting CPU, CUDA, and MPS.
- **High Performance**: Built on a multi-threaded C++ engine that outperforms `fitsio` and `astropy` by 2x-30x in typical workloads.
- **Smart Data Handling**: Stream large catalogs with predicate pushdown (`where="MAG < 20"`) and load massive images using memory-efficient chunking.
- **Astronomy Ready**: Full WCS support, efficient cutout reading, and Rice/HCOMPRESS handling out of the box.

## Install

```bash
pip install torchfits
```

## Quick Examples

### GPU-Accelerated Image Loading

Read science images directly to GPU memory without intermediate copies:

```python
import torchfits

# Load directly to CUDA device (or 'mps' on Mac)
data, header = torchfits.read(
    "science.fits",
    hdu=0,
    device='cuda',
    return_header=True
)
print(data.shape, data.dtype)  # torch.Size([4096, 4096]), torch.float32
```

### Efficient Catalog Filtering and Streaming

Filter million-row tables at the C++ level before loading into Python:

```python
# Read only standard stars brighter than mag 20
table = torchfits.table.read(
    "catalog.fits",
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20.0 AND CLASS_STAR > 0.9"
)

# Stream massive catalogs in batches
for batch in torchfits.table.scan("survey.fits", batch_size=50_000):
   process(batch)
```

## Performance Snapshot (v0.2.1)

TorchFits is designed to saturate modern IO subsystems, achieving a 100% win rate against standard baselines across our 88-case exhaustive benchmark suite. It delivers median speedups of **~2.5x** over `fitsio` and **>30x** over `astropy` for image I/O, while maintaining strict memory safety for massively large files. For machine learning pipelines, the custom `FITSDataset` and loader implementations match or exceed the throughput of optimized NumPy-based data loaders.

Full benchmarks: [`docs/benchmarks.md`](docs/benchmarks.md)

## Documentation

- User/API reference: [`docs/api.md`](docs/api.md)
- End-to-end examples: [`docs/examples.md`](docs/examples.md)
- Installation notes: [`docs/install.md`](docs/install.md)
- Changelog: [`docs/changelog.md`](docs/changelog.md)

## License

GPL-2.0
