"""
Minimal setup.py for TorchFits C++ extension only.
All project metadata is in pyproject.toml (modern approach).
"""

import os
import subprocess
from torch.utils.cpp_extension import BuildExtension, CppExtension

def get_wcslib_version():
    """Gets the WCSLIB version using pkg-config, or returns a default."""
    try:
        version_str = subprocess.check_output(
            ["pkg-config", "--modversion", "wcslib"],
            universal_newlines=True
        ).strip()
        major, minor = map(int, version_str.split(".")[:2])
        version_int = major * 1000000 + minor * 10000
        return version_int
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 7010000  # Default to 7.10

def get_cfitsio_include_path():
    """Finds the CFITSIO include path."""
    # Check for conda/pixi environment
    if 'CONDA_PREFIX' in os.environ:
        conda_path = os.path.join(os.environ['CONDA_PREFIX'], 'include')
        if os.path.exists(os.path.join(conda_path, 'fitsio.h')):
            return conda_path
    
    # Fallback paths
    possible_paths = [
        "/usr/include/cfitsio",
        "/usr/local/include/cfitsio", 
        "/opt/local/include/cfitsio",
    ]
    for path in possible_paths:
        if os.path.exists(os.path.join(path, "fitsio.h")):
            return path
    return "/usr/include/cfitsio"

def get_wcslib_include_path():
    """Finds the WCSLIB include path."""
    if 'CONDA_PREFIX' in os.environ:
        conda_path = os.path.join(os.environ['CONDA_PREFIX'], 'include')
        if os.path.exists(os.path.join(conda_path, 'wcslib')):
            return conda_path
    return "/usr/include/wcslib"

def get_cfitsio_library_path():
    """Finds the CFITSIO library path."""
    if 'CONDA_PREFIX' in os.environ:
        return os.path.join(os.environ['CONDA_PREFIX'], 'lib')
    return "/usr/lib"

def get_wcslib_library_path():
    """Finds the WCSLIB library path.""" 
    if 'CONDA_PREFIX' in os.environ:
        return os.path.join(os.environ['CONDA_PREFIX'], 'lib')
    return "/usr/lib"

# Build configuration
debug_mode = os.environ.get('DEBUG', '0') == '1'
extra_compile_args = ['-std=c++17', '-O2']  # Use O2 optimization
if debug_mode:
    extra_compile_args.extend(['-DDEBUG', '-g'])
else:
    extra_compile_args.extend(['-DNDEBUG'])  # Enable NDEBUG for release builds

# C++ Extension definition
ext_modules = [
    CppExtension(
        "torchfits.fits_reader_cpp",
        sources=[
            "src/torchfits/fits_reader.cpp",
            "src/torchfits/fits_utils.cpp", 
            "src/torchfits/wcs_utils.cpp",
            "src/torchfits/bindings.cpp",
            "src/torchfits/cache.cpp",
            "src/torchfits/remote.cpp",
            "src/torchfits/performance.cpp",
        ],
        include_dirs=[
            "src/torchfits",
            get_cfitsio_include_path(),
            get_wcslib_include_path()
        ],
        library_dirs=[
            get_cfitsio_library_path(),
            get_wcslib_library_path()
        ],
        libraries=["cfitsio", "wcs", "m", "curl"],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
]

# Minimal setup call - all metadata is in pyproject.toml
if __name__ == "__main__":
    from setuptools import setup
    setup(
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension},
    )
