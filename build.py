"""
Modern build configuration for torchfits C++ extensions.
This module is called by setuptools during the build process.
"""

import os
import sys
import platform
import subprocess

# Only import torch when needed, not at module level
def get_torch_extensions():
    """Import torch extensions only when needed."""
    try:
        from torch.utils.cpp_extension import CppExtension, BuildExtension
        return CppExtension, BuildExtension
    except ImportError:
        raise ImportError("PyTorch is required to build this package")

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

def get_pkg_config_paths(package, option):
    """Gets paths from pkg-config."""
    try:
        cmd = ["pkg-config", option, package]
        path_str = subprocess.check_output(cmd, universal_newlines=True).strip()
        # Remove -I or -L and split
        return [p[2:] for p in path_str.split()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

def get_cfitsio_include_paths():
    """Finds the CFITSIO include path."""
    paths = get_pkg_config_paths("cfitsio", "--cflags-only-I")
    if paths:
        return paths

    if 'CONDA_PREFIX' in os.environ:
        conda_path = os.path.join(os.environ['CONDA_PREFIX'], 'include')
        if os.path.exists(os.path.join(conda_path, 'fitsio.h')):
            return [conda_path]
    
    possible_paths = [
        "/usr/include/cfitsio",
        "/usr/local/include/cfitsio", 
        "/opt/local/include/cfitsio",
    ]
    for path in possible_paths:
        if os.path.exists(os.path.join(path, "fitsio.h")):
            return [path]
    return []

def get_wcslib_include_paths():
    """Finds the WCSLIB include path."""
    paths = get_pkg_config_paths("wcslib", "--cflags-only-I")
    if paths:
        return paths

    if 'CONDA_PREFIX' in os.environ:
        conda_path = os.path.join(os.environ['CONDA_PREFIX'], 'include')
        if os.path.exists(os.path.join(conda_path, 'wcslib')):
            return [conda_path]
    return []

def get_cfitsio_library_paths():
    """Finds the CFITSIO library path."""
    paths = get_pkg_config_paths("cfitsio", "--libs-only-L")
    if paths:
        return paths

    if 'CONDA_PREFIX' in os.environ:
        return [os.path.join(os.environ['CONDA_PREFIX'], 'lib')]
    return []

def get_wcslib_library_paths():
    """Finds the WCSLIB library path.""" 
    paths = get_pkg_config_paths("wcslib", "--libs-only-L")
    if paths:
        return paths

    if 'CONDA_PREFIX' in os.environ:
        return [os.path.join(os.environ['CONDA_PREFIX'], 'lib')]
    return []

def validate_dependencies():
    """Validate that all required dependencies are available."""
    issues = []
    
    cfitsio_includes = get_cfitsio_include_paths()
    if not cfitsio_includes:
        issues.append("CFITSIO not found. Please install it first.")
    
    wcslib_includes = get_wcslib_include_paths()
    if not wcslib_includes:
        issues.append("WCSLIB not found. Please install it first.")
    
    if issues:
        print("Error: Unable to build torchfits:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nSee README.md for installation instructions.")
        sys.exit(1)
    
    return cfitsio_includes, wcslib_includes

def get_platform_specific_settings():
    """Get platform-specific build settings."""
    libraries = ["cfitsio", "wcs", "m"]
    extra_compile_args = ['-std=c++17', '-O2']
    
    if platform.system() == "Windows":
        # Windows-specific settings
        libraries = ["cfitsio", "wcs"]
        extra_compile_args = ['/std:c++17', '/O2']
    elif platform.system() == "Darwin":
        # macOS-specific settings
        libraries.append("curl")
    else:
        # Linux and others
        libraries.append("curl")
    
    return libraries, extra_compile_args

def create_extension():
    """Create the C++ extension module."""
    CppExtension, BuildExtension = get_torch_extensions()
    
    # Validate dependencies first
    cfitsio_includes, wcslib_includes = validate_dependencies()
    
    # Get platform-specific settings
    libraries, platform_compile_args = get_platform_specific_settings()
    
    # Debug settings
    debug_mode = os.environ.get('DEBUG', '0') == '1'
    extra_compile_args = platform_compile_args.copy()
    
    if debug_mode:
        if platform.system() == "Windows":
            extra_compile_args.extend(['/DDEBUG', '/Z7'])
        else:
            extra_compile_args.extend(['-DDEBUG', '-g'])
    else:
        if platform.system() == "Windows":
            extra_compile_args.extend(['/DNDEBUG'])
        else:
            extra_compile_args.extend(['-DNDEBUG'])

    return CppExtension(
        "torchfits.fits_reader_cpp",
        sources=[
            "src/torchfits/fits_reader.cpp",
            "src/torchfits/fits_writer.cpp",  # New v1.0 writing functionality
            "src/torchfits/fits_utils.cpp", 
            "src/torchfits/wcs_utils.cpp",
            "src/torchfits/bindings.cpp",
            "src/torchfits/cache.cpp",
            "src/torchfits/real_cache.cpp",  # Real cache implementation
            "src/torchfits/cfitsio_enhanced.cpp",  # CFITSIO optimizations
            "src/torchfits/memory_optimizer.cpp",  # Memory-aligned tensor optimization
            "src/torchfits/fast_reader.cpp",  # FITSIO-inspired optimizations
            "src/torchfits/remote.cpp",
            # "src/torchfits/performance.cpp",  # Temporarily disabled for initial build
        ],
        include_dirs=[
            "src/torchfits",
            *cfitsio_includes,
            *wcslib_includes
        ],
        library_dirs=[
            *get_cfitsio_library_paths(),
            *get_wcslib_library_paths()
        ],
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        language="c++",
    )

# For backward compatibility and modern setuptools
def build(setup_kwargs=None):
    """
    Legacy function for setuptools compatibility.
    This is the main entry point called by setuptools.
    """
    if setup_kwargs is None:
        setup_kwargs = {}
        
    CppExtension, BuildExtension = get_torch_extensions()
    
    ext_modules = [create_extension()]
    
    setup_kwargs.update({
        "ext_modules": ext_modules,
        "cmdclass": {
            "build_ext": BuildExtension
        }
    })
    return setup_kwargs
