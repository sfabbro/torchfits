# Environment & Platform Compatibility

Supported Python versions, PyTorch runtime ABIs, CUDA flavors, and operating systems for **torchfits**.

For supported FITS formats, HDU types, tile compression algorithms, and catalog features, see the [Feature Parity Matrix](parity.md).

---

## Supported Environments

| Component | Prebuilt Wheels | Source Builds |
|---|---|---|
| **Python** | **3.10, 3.11, 3.12, 3.13, 3.14** | **3.10+** |
| **PyTorch** | **2.13.x** (default on PyPI)<br>**2.12.x, 2.11.x** (GitHub Releases) | **≥ 2.10** (`pip install --no-deps --no-build-isolation .`) |
| **Hardware & CUDA** | **CPU**, **CUDA 12.6, 12.8, 12.9, 13.0**, **Apple Silicon MPS** | All PyTorch-supported compute devices |
| **Operating Systems** | **Linux** (`x86_64`, `aarch64`)<br>**macOS** (`arm64` Apple Silicon) | Linux, macOS |
| **Core Libraries** | **NumPy ≥ 1.20**, **PyArrow ≥ 5.0** | Same |

---

## PyTorch Minor Version ABI Matching

Because PyTorch does not guarantee C++ ABI stability across minor version releases ($2.11 \to 2.12 \to 2.13$), each `torchfits` binary wheel embeds the specific PyTorch C++ ABI tag it was compiled against.

| PyTorch Version | Wheel Distribution Channel | Installation Command |
|---|---|---|
| **PyTorch 2.13.x** | **Default PyPI Release** | `pip install torchfits` |
| **PyTorch 2.12.x** | **GitHub Release Wheels** | `pip install https://github.com/astroai/torchfits/releases/download/v1.0.0/torchfits-1.0.0+torch212-cp311-cp311-manylinux_2_28_x86_64.whl` |
| **PyTorch 2.11.x** | **GitHub Release Wheels** | `pip install https://github.com/astroai/torchfits/releases/download/v1.0.0/torchfits-1.0.0+torch211-cp311-cp311-manylinux_2_28_x86_64.whl` |
| **PyTorch 2.10.x** | **Source Build** | `pip install --no-deps --no-build-isolation .` |

---

## CUDA & Accelerator Compatibility

- **Universal CUDA / CPU Wheels:** A single `torchfits` wheel functions across all CUDA flavors (`cu126`, `cu128`, `cu129`, `cu130`) as well as CPU-only (`+cpu`) installations of the matching PyTorch minor version.
- **Apple Silicon (MPS):** Native `arm64` wheels for macOS leverage Metal Performance Shaders (`device="mps"`).
- **Graceful Fallback:** CUDA-built environments run seamlessly on CPU-only machines via automatic CPU fallback.

---

## Verification

To verify that your installation matches your current Python and PyTorch runtime:

```python
import torch
import torchfits

print("torchfits version:", torchfits.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA GPU available:", torch.cuda.is_available())
print("Apple MPS available:", torch.backends.mps.is_available())
```

