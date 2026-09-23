# Environment & Platform Compatibility

Supported Python versions, PyTorch runtime ABIs, CUDA flavors, and operating systems for **torchfits**.

For supported FITS formats, HDU types, tile compression algorithms, and catalog features, see the [Feature Parity Matrix](parity.md).

---

## Supported Environments

| Component | Prebuilt Wheels | Source Builds |
|---|---|---|
| **Python** | **3.10, 3.11, 3.12, 3.13, 3.14** | **3.10+** |
| **PyTorch** | **2.13.x** (ABI-matched wheels) | **≥ 2.10** (`pip install --no-deps --no-build-isolation .`) |
| **Hardware & CUDA** | **CPU**, **CUDA 12.6, 12.9, 13.0**, **Apple Silicon MPS** | All PyTorch-supported compute devices |
| **Operating Systems** | **Linux** (`x86_64`, `aarch64`)<br>**macOS** (`arm64` Apple Silicon) | Linux, macOS |
| **Core Libraries** | **NumPy ≥ 1.20**, **PyArrow ≥ 5.0** | Same |

---

## PyTorch Minor Version ABI Matching

Because PyTorch does not guarantee C++ ABI stability across minor version releases ($2.11 \to 2.12 \to 2.13$), each `torchfits` binary wheel embeds the specific PyTorch C++ ABI tag it was compiled against.

| PyTorch Version | Wheel Distribution Channel | Installation Command |
|---|---|---|
| **PyTorch 2.13.x** | **Default PyPI Release** | `pip install torchfits` |
| **Any other minor (≥ 2.10)** | **Source Build** | `pip install --no-deps --no-build-isolation .` |

---

## CUDA & Accelerator Compatibility

- **Universal CUDA / CPU Wheels:** A single `torchfits` wheel functions across all CUDA flavors of its PyTorch minor version (`cu126`, `cu129`, `cu130`) as well as CPU-only (`+cpu`) installations.
- **Apple Silicon (MPS):** Native `arm64` wheels for macOS leverage Metal Performance Shaders (`device="mps"`).
- **Graceful Fallback:** CUDA-built environments run seamlessly on CPU-only machines via automatic CPU fallback.
- **MPS dtype handling:** `device="mps"` (and `mps:N`) downcasts `float64 → float32` and `complex128 → complex64` before the host-to-device transfer because MPS has no native 64-bit float/complex. Each downcast emits a `UserWarning` (Python's default filter shows it once per call site): `MPS does not support float64; downcasting to float32 (precision loss)` and `MPS does not support complex128; downcasting to complex64 (precision loss)`. The result keeps its shape and lands on the requested device; CPU and CUDA paths keep 64-bit.
- **Scale precision note:** Image BSCALE/BZERO scaling is applied in `float32` (`read_full_scaled_cpu`), while table `TSCAL/TZERO` scaling uses `float64`. No divergence vs astropy has been observed for integer storage, but fractional-scaled `LONGLONG` (`BITPIX=64`) images lose precision relative to the table path. A `float64` accumulation for images is planned for 2.0.

### Known limitations

- **No Windows support:** Prebuilt wheels and CI are Linux and macOS only; Windows is documented as unsupported.
- **Runtime env knobs:** every `TORCHFITS_*` variable the library reads, with its default and effect, is listed under [Environment variables](architecture.md#environment-variables). Those knobs are not a stable public API and may change without notice.
- **`KMP_DUPLICATE_LIB_OK`:** importing `torchfits` runs `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` before PyTorch loads. macOS otherwise aborts when PyTorch's `libomp` and a second OpenMP runtime (Homebrew or conda) are both mapped. An existing value is left alone. See [Install](install.md) if you import `torch` first.
- **CFITSIO DNS:** `http`/`https`/`ftp` paths are classified at guard time. Python fetches (`http_open`) dial the addresses from that lookup. A URL that CFITSIO opens itself is only checked then; the driver re-resolves the host afterwards (residual TOCTOU). Pinning those connections would need a custom CFITSIO driver and is not implemented.
- **Legacy knobs:** `ReadOptions.handle_cache_capacity` and `clear_file_cache(handles=)` are deprecated no-ops retained for compatibility; the unified `SharedReadMeta` cache is the live shared cache. They will be removed in 2.0 (deprecation warning now).

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
