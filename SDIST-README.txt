torchfits source distribution
=============================

This sdist is provided for source inspection and packager review only.

Building it from source requires a pre-installed PyTorch (>= 2.10, ABI-matched),
a C++17 toolchain, and network access to fetch the pinned CFITSIO sources
(extern/vendor.sh --cfitsio-version extern/VERSIONS.txt). It is NOT built or
tested as an install path: PyPI installs torchfits from prebuilt wheels.

If you need a source build, clone the git repository instead — the CMake layer
auto-vendors the pinned CFITSIO there.
