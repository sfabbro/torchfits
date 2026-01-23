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

message(STATUS "DEBUG: Python_EXECUTABLE: ${Python_EXECUTABLE}")
message(STATUS "DEBUG: Python_VERSION: ${Python_VERSION}")
message(STATUS "DEBUG: NANOBIND_PYTHON_EXECUTABLE: ${NANOBIND_PYTHON_EXECUTABLE}")
message(STATUS "DEBUG: nanobind_DIR: ${nanobind_DIR}")
message(STATUS "DEBUG: nanobind_ROOT: ${nanobind_ROOT}")

# Find PyTorch
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

# Find ZLIB (Required for compressed FITS support)
find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
    message(STATUS "Found ZLIB: ${ZLIB_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "ZLIB not found - required for compressed FITS support")
endif()

# Vendor CFITSIO
get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/../../.." ABSOLUTE)
set(CFITSIO_SOURCE_DIR "${PROJECT_ROOT}/extern/cfitsio")

if(EXISTS "${CFITSIO_SOURCE_DIR}/CMakeLists.txt")
    message(STATUS "Building vendored CFITSIO from ${CFITSIO_SOURCE_DIR}")
    
    # CFITSIO Build Options
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
    set(USE_CURL OFF CACHE BOOL "Disable CURL for now" FORCE)
    set(TESTS OFF CACHE BOOL "Disable tests" FORCE)
    set(UTILS OFF CACHE BOOL "Disable utils" FORCE)
    set(USE_PTHREADS ON CACHE BOOL "Enable thread-safe build" FORCE)
    
    # Force position-independent code for static library to be linked into shared library
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    
    add_subdirectory("${CFITSIO_SOURCE_DIR}" cfitsio_build)
    
    # CFITSIO target is named 'cfitsio'
    set(CFITSIO_LIBRARY cfitsio)
    set(CFITSIO_INCLUDE_DIR "${CFITSIO_SOURCE_DIR}") # For direct include if needed
else()
    message(FATAL_ERROR "Vendored CFITSIO not found at ${CFITSIO_SOURCE_DIR}. Please run 'git submodule update --init --recursive'")
endif()

# Find DLPack headers (used for zero-copy tensor exchange)
find_path(DLPACK_INCLUDE_DIR dlpack/dlpack.h)
if(DLPACK_INCLUDE_DIR)
    message(STATUS "Found DLPack include: ${DLPACK_INCLUDE_DIR}")
else()
    message(WARNING "DLPack headers not found - DLPack-based zero-copy will fail to compile until headers are available")
endif()

# Find WCSLIB
if(PKG_CONFIG_FOUND)
    pkg_check_modules(WCSLIB QUIET wcslib)
endif()

if(NOT WCSLIB_FOUND)
    find_path(WCSLIB_INCLUDE_DIR wcslib/wcs.h)
    find_library(WCSLIB_LIBRARY wcs)
    if(WCSLIB_INCLUDE_DIR AND WCSLIB_LIBRARY)
        set(WCSLIB_FOUND TRUE)
    endif()
endif()

# Source files
set(SOURCES
    bindings.cpp
)

# Create python binding module
nanobind_add_module(cpp ${SOURCES})

# Link libraries
target_link_libraries(cpp PRIVATE ${TORCH_LIBRARIES})

# Link torch_python for THPVariable_Wrap (bypasses DLPack overhead)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch, os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "DEBUG: TORCH_ROOT: ${TORCH_ROOT}")

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH)
if(TORCH_PYTHON_LIBRARY)
    target_link_libraries(cpp PRIVATE ${TORCH_PYTHON_LIBRARY})
    message(STATUS "Found libtorch_python: ${TORCH_PYTHON_LIBRARY}")
else()
    message(WARNING "libtorch_python not found in ${TORCH_ROOT}/lib - tensor returns will use slower DLPack path")
    # Try finding it in site-packages/torch/lib as fallback
    find_library(TORCH_PYTHON_LIBRARY_FALLBACK torch_python PATHS "${PYTHON_SITE_PACKAGES}/torch/lib" NO_DEFAULT_PATH)
    if(TORCH_PYTHON_LIBRARY_FALLBACK)
        target_link_libraries(cpp PRIVATE ${TORCH_PYTHON_LIBRARY_FALLBACK})
        message(STATUS "Found libtorch_python fallback: ${TORCH_PYTHON_LIBRARY_FALLBACK}")
    endif()
endif()

if(TARGET cfitsio)
    target_link_libraries(cpp PRIVATE cfitsio)
    target_include_directories(cpp PRIVATE "${CFITSIO_INCLUDE_DIR}")
else()
    target_link_libraries(cpp PRIVATE ${CFITSIO_LIBRARIES} ${CFITSIO_LIBRARY})
    target_include_directories(cpp PRIVATE ${CFITSIO_INCLUDE_DIRS} ${CFITSIO_INCLUDE_DIR})
endif()
target_link_libraries(cpp PRIVATE ZLIB::ZLIB)

if(DLPACK_INCLUDE_DIR)
    target_include_directories(cpp PRIVATE ${DLPACK_INCLUDE_DIR})
endif()
target_compile_definitions(cpp PRIVATE HAS_CFITSIO)

if(WCSLIB_LIBRARY)
    target_link_libraries(cpp PRIVATE ${WCSLIB_LIBRARY})
    target_include_directories(cpp PRIVATE ${WCSLIB_INCLUDE_DIR})
    target_compile_definitions(cpp PRIVATE HAS_WCSLIB)
endif()

# Compiler flags
target_compile_definitions(cpp PRIVATE VERSION_INFO="")
target_compile_features(cpp PRIVATE cxx_std_17)

# Compiler optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -funroll-loops)
endif()

# Enable SIMD optimizations for hardware acceleration
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_definitions(cpp PRIVATE CFITSIO_HAVE_SSSE3)
    target_compile_options(cpp PRIVATE -march=native)
    message(STATUS "Enabled Native x86_64 optimizations")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(cpp PRIVATE -mcpu=native)
    message(STATUS "Enabled Native ARM64 optimizations")
endif()

# Install the compiled module
# For scikit-build-core, install to the package directory
install(TARGETS cpp DESTINATION torchfits)