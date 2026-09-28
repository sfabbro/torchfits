torchfits source distribution
=============================

This sdist is provided for source inspection and packager review only.

Building it from source requires a C++17 toolchain and network access to fetch
the pinned CFITSIO sources (extern/vendor.sh --cfitsio-version
extern/VERSIONS.txt). PyTorch must be the exact minor this sdist is built for,
not merely a minimum version: the CMake layer stamps the build-time PyTorch
major.minor into the extension as its ABI, and the extension refuses to import
against any other minor. The pin is the torch entry under
`build-system.requires` in pyproject.toml; with a matching PyTorch already
installed, build with --no-build-isolation. It is NOT built or tested as an
install path: PyPI installs torchfits from prebuilt wheels.

If you need a source build, clone the git repository instead — the CMake layer
auto-vendors the pinned CFITSIO there.
