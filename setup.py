# setup.py
import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

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
    # If not found, return empty
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

class CustomBuildExt(build_ext):
    """Custom build_ext command to dynamically set WCSLIB_VERSION and include/lib paths."""

    def build_extensions(self):
        wcslib_version = get_wcslib_version()
        cfitsio_include_dir = get_cfitsio_include_path()
        cfitsio_library_dir = get_cfitsio_library_path()
        wcslib_include_dir = get_wcslib_include_path()
        wcslib_library_dir = get_wcslib_library_path()


        for ext in self.extensions:
            ext.extra_compile_args.append(f"-DWCSLIB_VERSION={wcslib_version}")
            if cfitsio_include_dir:
                ext.include_dirs.append(cfitsio_include_dir)
            if wcslib_include_dir:
                ext.include_dirs.append(wcslib_include_dir)
            if cfitsio_library_dir:
                ext.library_dirs.append(cfitsio_library_dir)
            if wcslib_library_dir:
                ext.library_dirs.append(wcslib_library_dir)

        super().build_extensions()


setup(
    cmdclass={'build_ext': CustomBuildExt},
)