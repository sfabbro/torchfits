# Environment & Platform Compatibility

Supported Python versions, PyTorch runtime ABIs, CUDA flavors, and operating systems for **torchfits**.

For supported FITS formats, HDU types, tile compression, and catalog features, see [Evaluation](evaluation.md).

---

## Supported Environments

| Component | Prebuilt Wheels | Source Builds |
|---|---|---|
| **Python** | **3.11, 3.12, 3.13, 3.14**; **3.15** (Linux only) | **3.11+** |
| **PyTorch** | **2.14.x** (ABI-matched wheels) | **≥ 2.10** (`pip install --no-deps --no-build-isolation .`) |
| **Hardware & CUDA** | **CPU**, **CUDA 12.6, 13.0**, **Apple Silicon MPS** | All PyTorch-supported compute devices |
| **Operating Systems** | **Linux** (`x86_64`, `aarch64`)<br>**macOS** (`arm64` Apple Silicon) | Linux, macOS |
| **Core Libraries** | **NumPy ≥ 1.26**, **PyArrow ≥ 17.0** | Same |

---

## PyTorch Minor Version ABI Matching

Because PyTorch does not guarantee C++ ABI stability across minor version releases ($2.12 \to 2.13 \to 2.14$), each `torchfits` binary wheel embeds the specific PyTorch C++ ABI tag it was compiled against.

| PyTorch Version | Wheel Distribution Channel | Installation Command |
|---|---|---|
| **PyTorch 2.14.x** | **Default PyPI Release** | `pip install torchfits` |
| **Any other minor (≥ 2.10)** | **Source Build** | `pip install --no-deps --no-build-isolation .` |

---

## CUDA & Accelerator Compatibility

- **Universal CUDA / CPU Wheels:** A single `torchfits` wheel functions across all CUDA flavors of its PyTorch minor version (`cu126`, `cu130`) as well as CPU-only (`+cpu`) installations.
- **Apple Silicon (MPS):** Native `arm64` wheels for macOS leverage Metal Performance Shaders (`device="mps"`).
- **Graceful Fallback:** CUDA-built environments run seamlessly on CPU-only machines via automatic CPU fallback.
- **MPS dtype handling:** `device="mps"` (and `mps:N`) downcasts `float64 → float32` and `complex128 → complex64` before the host-to-device transfer because MPS has no native 64-bit float/complex. Each downcast emits a `UserWarning` (Python's default filter shows it once per call site): `MPS does not support float64; downcasting to float32 (precision loss)` and `MPS does not support complex128; downcasting to complex64 (precision loss)`. The result keeps its shape and lands on the requested device; CPU and CUDA paths keep 64-bit.
- **Scaled images:** A real `BSCALE`/`BZERO` on BITPIX=8 or 16 is read as float32. The same on BITPIX=32 or 64 is read as float64, so integers above 2^24 are not rounded. `BLANK` on a plain integer image uses that same float dtype so the blank pixel can be NaN. Signed-byte and unsigned integer conventions (`BZERO` of -128, 32768, or 2147483648 with `BSCALE=1`) stay integers. Native `BITPIX=-32` images stay float32. Table `TSCAL`/`TZERO` on real columns is float64.

### Known limitations

- **No Python 3.15 macOS wheel in 1.3.0:** PyTorch 2.14's cp315 macOS and Windows wheels were built against CPython 3.15.0b3 headers, whose `PyThreadState` layout predates 3.15.0b4. The first Python-level tensor operation outside the main thread then segfaults inside pybind11's `gil_scoped_acquire` ([pytorch/pytorch#200779](https://github.com/pytorch/pytorch/issues/200779)); the Linux cp315 wheels are unaffected. `torchfits` 1.3.0 therefore publishes cp315 wheels for Linux only; macOS users on Python 3.15 need PyTorch 2.15, which this release does not target.
- **No Windows support:** Prebuilt wheels and CI are Linux and macOS only; Windows is documented as unsupported.
- **Runtime env knobs:** every `TORCHFITS_*` variable the library reads, with its default and effect, is listed under [Environment variables](architecture.md#environment-variables). Those knobs are not a stable public API and may change without notice.
- **`KMP_DUPLICATE_LIB_OK`:** importing `torchfits` runs `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` before PyTorch loads. macOS otherwise aborts when PyTorch's `libomp` and a second OpenMP runtime (Homebrew or conda) are both mapped. An existing value is left alone. See [Install](install.md) if you import `torch` first.
- **CFITSIO DNS:** `http`/`https`/`ftp` paths are classified at guard time. Python fetches (`http_open`) dial the addresses from that lookup. A URL that CFITSIO opens itself is only checked then; the driver re-resolves the host afterwards (residual TOCTOU). Pinning those connections would need a custom CFITSIO driver and is not implemented.
- **Ignored arguments:** `cache_capacity`, `handle_cache_capacity`, and `fast_header` on a table read do nothing. So does `handles=` on `clear_file_cache()`. Leave them out of new code.

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
