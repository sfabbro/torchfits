cmake_minimum_required(VERSION 3.15)
project(torchfits_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find nanobind
# Detect the installed nanobind package and import it into CMake
# For pixi builds, nanobind is in the build environment, not host environment
if(DEFINED ENV{BUILD_PREFIX})
    # In pixi build environment, use build prefix python for nanobind detection
    set(NANOBIND_PYTHON_EXECUTABLE "$ENV{BUILD_PREFIX}/bin/python")
    message(STATUS "Using build environment Python for nanobind: ${NANOBIND_PYTHON_EXECUTABLE}")
else()
    # Normal development environment
    set(NANOBIND_PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
endif()

execute_process(
  COMMAND "${NANOBIND_PYTHON_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
  ERROR_QUIET
)

if(nanobind_ROOT)
    message(STATUS "Found nanobind cmake directory: ${nanobind_ROOT}")
    list(APPEND CMAKE_PREFIX_PATH ${nanobind_ROOT})
else()
    # Fallback: try to find nanobind via Python import in build environment
    execute_process(
        COMMAND "${NANOBIND_PYTHON_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
        ERROR_QUIET
    )
    if(nanobind_ROOT)
        message(STATUS "Found nanobind via Python import: ${nanobind_ROOT}")
        list(APPEND CMAKE_PREFIX_PATH ${nanobind_ROOT})
    endif()
endif()

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
) 
find_package(nanobind CONFIG REQUIRED)

# Find PyTorch
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

# Find CFITSIO with performance flags
find_path(CFITSIO_INCLUDE_DIR fitsio.h)
find_library(CFITSIO_LIBRARY cfitsio)

# Find DLPack headers (used for zero-copy tensor exchange)
find_path(DLPACK_INCLUDE_DIR dlpack/dlpack.h)
if(DLPACK_INCLUDE_DIR)
    message(STATUS "Found DLPack include: ${DLPACK_INCLUDE_DIR}")
else()
    message(WARNING "DLPack headers not found - DLPack-based zero-copy will fail to compile until headers are available")
endif()

# Check if CFITSIO has thread safety and hardware acceleration
include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES ${CFITSIO_LIBRARY})
set(CMAKE_REQUIRED_INCLUDES ${CFITSIO_INCLUDE_DIR})

check_c_source_compiles("
#include <fitsio.h>
int main() {
    int status;
    fits_is_reentrant(&status);
    return 0;
}
" CFITSIO_HAS_REENTRANT)

if(CFITSIO_HAS_REENTRANT)
    message(STATUS "CFITSIO has thread safety enabled")
    target_compile_definitions(cpp PRIVATE CFITSIO_THREAD_SAFE)
else()
    message(WARNING "CFITSIO lacks thread safety - performance may be limited")
endif()

# Find WCSLIB
find_path(WCSLIB_INCLUDE_DIR wcslib/wcs.h)
find_library(WCSLIB_LIBRARY wcs)

# Source files
set(SOURCES
    bindings.cpp
)

# Create python binding module
nanobind_add_module(cpp ${SOURCES})

# Link libraries
target_link_libraries(cpp PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(cpp PRIVATE ${CFITSIO_LIBRARY})
target_include_directories(cpp PRIVATE ${CFITSIO_INCLUDE_DIR})
if(DLPACK_INCLUDE_DIR)
    target_include_directories(cpp PRIVATE ${DLPACK_INCLUDE_DIR})
endif()
target_compile_definitions(cpp PRIVATE HAS_CFITSIO)

target_link_libraries(cpp PRIVATE ${WCSLIB_LIBRARY})
target_include_directories(cpp PRIVATE ${WCSLIB_INCLUDE_DIR})
target_compile_definitions(cpp PRIVATE HAS_WCSLIB)

# Compiler flags
target_compile_definitions(cpp PRIVATE VERSION_INFO="")
target_compile_features(cpp PRIVATE cxx_std_17)

# Fix PyTorch linking issues
set_target_properties(cpp PROPERTIES
    CXX_VISIBILITY_PRESET "default"
    VISIBILITY_INLINES_HIDDEN OFF
)

# OpenMP for parallel processing
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cpp PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(cpp PRIVATE HAS_OPENMP)
endif()

# Enable SIMD optimizations for hardware acceleration
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_definitions(cpp PRIVATE CFITSIO_HAVE_SSSE3)
    target_compile_options(cpp PRIVATE -mssse3)
    message(STATUS "Enabled SSSE3 optimizations for x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(cpp PRIVATE -march=armv8-a)
    message(STATUS "Enabled ARM64 optimizations")
endif()

# Install the compiled module
# For pixi-build, install to the site-packages/torchfits directory
install(TARGETS cpp DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/torchfits)
