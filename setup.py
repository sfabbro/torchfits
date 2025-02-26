import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

debug_mode = os.environ.get('DEBUG', '0') == '1'

extra_compile_args = ['-std=c++17']
if debug_mode:
    extra_compile_args.extend(['-DDEBUG', '-g'])


def get_wcslib_version():
    """Gets the WCSLIB version using pkg-config, or returns a default."""
    try:
        version_str = subprocess.check_output(
            ["pkg-config", "--modversion", "wcslib"],
            universal_newlines=True
        ).strip()
        # Convert version string to integer (e.g., "7.10" -> 7010000)
        major, minor = map(int, version_str.split(".")[:2])
        version_int = major * 1000000 + minor * 10000
        print(f"Detected WCSLIB version: {version_str} (int: {version_int})")
        return version_int
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Could not detect WCSLIB version with pkg-config. Using default (7.10).")
        return 7010000  # Default to 7.10

def get_cfitsio_include_path():
    """Finds the CFITSIO include path, checking common locations."""
    # Check common installation paths
    possible_paths = [
        "/usr/include/cfitsio",
        "/usr/local/include/cfitsio",
        "/opt/local/include/cfitsio",  # MacPorts
        "/sw/include/cfitsio",         # Fink
    ]
    for path in possible_paths:
        if os.path.exists(os.path.join(path, "fitsio.h")):
            return path

    # Try using pkg-config (if available)
    try:
        include_path = subprocess.check_output(
            ["pkg-config", "--cflags-only-I", "cfitsio"],
            universal_newlines=True
        ).strip()
        #Remove the -I
        include_path = include_path.replace("-I","")
        if os.path.exists(os.path.join(include_path, "fitsio.h")):
            return include_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # If not found, return empty string (user will need to specify)
    print("Warning: Could not automatically find CFITSIO include path.")
    return ""

def get_cfitsio_library_path():
    """Find cfitsio library using pkg-config"""
    try:
        library_path = subprocess.check_output(
            ["pkg-config", "--libs-only-L", "cfitsio"],
            universal_newlines=True
        ).strip()
         #Remove the -L
        library_path = library_path.replace("-L","")
        return library_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

     # Check common installation paths
    possible_paths = [
        "/usr/lib",
        "/usr/local/lib",
        "/opt/local/lib",
        "/sw/lib",
        "/usr/lib/x86_64-linux-gnu" #Debian/Ubuntu specific path
    ]
    for path in possible_paths:
        if (os.path.exists(os.path.join(path, "libcfitsio.so")) or
           os.path.exists(os.path.join(path, "libcfitsio.dylib")) or #macOS
           os.path.exists(os.path.join(path, "cfitsio.lib"))):  #Windows
            return path
    # If not found, return empty string
    print("Warning: Could not automatically find CFITSIO library path.")
    return ""


def get_wcslib_include_path():
    """Finds the wcslib include path, checking common locations."""
    possible_paths = [
        "/usr/include",
        "/usr/local/include",
        "/opt/local/include",  # MacPorts
        "/sw/include",         # Fink
    ]
    for path in possible_paths:
        if os.path.exists(os.path.join(path, "wcslib", "wcs.h")):
            return os.path.join(path, "wcslib")  # Include the wcslib subdirectory

    # Try using pkg-config (if available)
    try:
        include_path = subprocess.check_output(
            ["pkg-config", "--cflags-only-I", "wcslib"],
            universal_newlines=True
        ).strip()
        include_path = include_path.replace("-I","")
        if os.path.exists(os.path.join(include_path, "wcs.h")):
            return include_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    # If not found, return empty string
    print("Warning: Could not automatically find WCSLIB include path.")
    return ""

def get_wcslib_library_path():
    """Find wcslib library using pkg-config."""

    try:
        library_path = subprocess.check_output(
            ["pkg-config", "--libs-only-L", "wcslib"],
            universal_newlines=True
        ).strip()
         #Remove the -L
        library_path = library_path.replace("-L","")
        return library_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

     # Check common installation paths
    possible_paths = [
        "/usr/lib",
        "/usr/local/lib",
        "/opt/local/lib",
        "/sw/lib",
        "/usr/lib/x86_64-linux-gnu"
    ]
    for path in possible_paths:
        if (os.path.exists(os.path.join(path, "libwcs.so")) or  # Linux
           os.path.exists(os.path.join(path, "libwcs.dylib")) or #macOS
           os.path.exists(os.path.join(path, "wcs.lib"))):  #Windows
            return path
    # If not found, return empty string
    print("Warning: Could not automatically find WCSLIB library path.")
    return ""




# --- Extension Definition ---
ext_modules = [
    CppExtension(
        "torchfits.fits_reader",
        sources=[
            "src/torchfits/fits_reader.cpp",
            "src/torchfits/fits_utils.cpp",
            "src/torchfits/wcs_utils.cpp",
            "src/torchfits/bindings.cpp"
        ],
        include_dirs=["src/torchfits"],
        libraries=["cfitsio", "wcs", "m"],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)
