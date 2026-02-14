# Installation Guide

`torchfits` can be installed via pip or built from source.

## Standard Installation (Pip)

The easiest way to install `torchfits` is using `pip`:

```bash
pip install torchfits
```

This will download and install the latest stable version from PyPI.

## Building from Source

To build `torchfits` from source, you'll need a C++ compiler and CMake.

### Prerequisites

- Python 3.11+
- C++17 compatible compiler (GCC, Clang, MSVC)
- CMake 3.21+
- Ninja (recommended)

### Steps

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/sfabbro/torchfits.git
    cd torchfits
    ```

2.  **Vendor Dependencies:**

    `torchfits` vendors its C dependencies (`cfitsio` and `wcslib`) to simplify installation. Run the vendoring script to prepare the source tree:

    ```bash
    ./extern/vendor.sh
    ```

    This script downloads the required versions of `cfitsio` and `wcslib` and places them in the `extern/` directory.
    It resolves latest tags by default; pin versions with `--cfitsio-version` and `--wcslib-version` for reproducible builds.

3.  **Install:**

    You can then install the package using `pip` in editable mode for development:

    ```bash
    pip install -e .
    ```

    Or build a release version:

    ```bash
    pip install .
    ```

### Troubleshooting

- **Missing `wcslib` or `cfitsio`?**
  Ensure you ran `./extern/vendor.sh` before installing. The build system expects these sources to be present in `extern/`.

- **Compiler Errors?**
  Verify your C++ compiler supports C++17. On macOS, ensure Xcode Command Line Tools are installed (`xcode-select --install`).
