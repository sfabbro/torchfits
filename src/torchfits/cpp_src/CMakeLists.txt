cmake_minimum_required(VERSION 3.15)
project(torchfits_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Global optimizations (Release mode)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Using only -O3 (avoid -funroll-loops as it may conflict with manual SIMD unrolling)
    add_compile_options(-O3)
endif()

# Enable IPO/LTO when available (portable opt-in)
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if(IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
else()
    message(STATUS "IPO/LTO not supported: ${IPO_ERROR}")
endif()

# Enable SIMD optimizations globally (propagates to cfitsio)
# Note: For wheels, avoid -march=native. Using reasonable baselines.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # Enable SSSE3 for Rice compression optimization (reasonable baseline for modern PyTorch)
    add_compile_definitions(CFITSIO_HAVE_SSSE3)
    add_compile_options(-mssse3) 
    message(STATUS "Enabled x86_64 optimizations (SSSE3)")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    # ARM64 implies NEON, -O3 handles vectorization well.
    # On Apple Silicon, -mcpu=native ensures correct scheduling for M-series.
    # This assumes building on M1+ for M1+ target (standard for macOS wheels).
    if(APPLE)
       message(STATUS "Enabled Native Apple Silicon optimizations")
    else()
       message(STATUS "Enabled ARM64 optimizations")
    endif()
endif()


# Add environment prefix to search path (important for pixi/conda)
if(DEFINED ENV{PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{PREFIX}")
    message(STATUS "Added environment prefix to CMAKE_PREFIX_PATH: $ENV{PREFIX}")
endif()

# Find required packages
# In pixi/conda packaging builds, ensure Python comes from the host prefix.
if(DEFINED ENV{PREFIX} AND EXISTS "$ENV{PREFIX}/bin/python")
    set(Python_ROOT_DIR "$ENV{PREFIX}" CACHE PATH "Python root directory" FORCE)
    set(Python_EXECUTABLE "$ENV{PREFIX}/bin/python" CACHE FILEPATH "Python executable" FORCE)
endif()
set(Python_FIND_STRATEGY LOCATION)
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

# Some distributions ship a Torch CMake target with include paths that may be missing.
# Filter out any non-existent include dirs to avoid CMake configure errors.
if(TARGET torch)
    get_target_property(TORCH_INCLUDE_DIRS torch INTERFACE_INCLUDE_DIRECTORIES)
    if(TORCH_INCLUDE_DIRS)
        set(TORCH_INCLUDE_DIRS_FILTERED "")
        foreach(DIR ${TORCH_INCLUDE_DIRS})
            if(EXISTS "${DIR}")
                list(APPEND TORCH_INCLUDE_DIRS_FILTERED "${DIR}")
            else()
                message(STATUS "Dropping missing Torch include dir: ${DIR}")
            endif()
        endforeach()
        if(TORCH_INCLUDE_DIRS_FILTERED)
            set_target_properties(torch PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS_FILTERED}")
        endif()
    endif()
endif()

# Find ZLIB (Required for compressed FITS support)
find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
    message(STATUS "Found ZLIB: ${ZLIB_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "ZLIB not found - required for compressed FITS support")
endif()

get_filename_component(TORCHFITS_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(CFITSIO_SOURCE_DIR "${TORCHFITS_REPO_ROOT}/extern/cfitsio")
set(WCSLIB_SOURCE_DIR "${TORCHFITS_REPO_ROOT}/extern/wcslib")
set(TORCHFITS_VENDOR_SCRIPT "${TORCHFITS_REPO_ROOT}/extern/vendor.sh")
option(TORCHFITS_AUTO_VENDOR_DEPS "Auto-run extern/vendor.sh when vendored deps are missing" ON)
set(TORCHFITS_VENDOR_RAN FALSE)

macro(torchfits_run_vendor_once)
    if(TORCHFITS_AUTO_VENDOR_DEPS AND NOT TORCHFITS_VENDOR_RAN)
        if(EXISTS "${TORCHFITS_VENDOR_SCRIPT}")
            message(STATUS "Vendored deps missing; running ${TORCHFITS_VENDOR_SCRIPT}")
            execute_process(
                COMMAND bash "${TORCHFITS_VENDOR_SCRIPT}"
                WORKING_DIRECTORY "${TORCHFITS_REPO_ROOT}"
                RESULT_VARIABLE TORCHFITS_VENDOR_RESULT
            )
            if(NOT TORCHFITS_VENDOR_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to run ${TORCHFITS_VENDOR_SCRIPT} (exit ${TORCHFITS_VENDOR_RESULT})")
            endif()
            set(TORCHFITS_VENDOR_RAN TRUE)
        endif()
    endif()
endmacro()

# Find CFITSIO
option(TORCHFITS_USE_VENDORED_CFITSIO "Prefer vendored CFITSIO over environment" ON)
set(TORCHFITS_NIOBUF "" CACHE STRING "Override CFITSIO NIOBUF (default 40)")
set(TORCHFITS_MINDIRECT "" CACHE STRING "Override CFITSIO MINDIRECT (default 8640)")

if(TORCHFITS_USE_VENDORED_CFITSIO)
    unset(CFITSIO_FOUND CACHE)
    unset(CFITSIO_INCLUDE_DIR CACHE)
    unset(CFITSIO_LIBRARY CACHE)
    unset(CFITSIO_INCLUDE_DIRS CACHE)
    unset(CFITSIO_LIBRARIES CACHE)
    unset(CFITSIO_LIBRARY_DIRS CACHE)
    set(CFITSIO_FOUND FALSE)
endif()

if(NOT TORCHFITS_USE_VENDORED_CFITSIO)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(CFITSIO QUIET cfitsio)
    endif()

    if(NOT CFITSIO_FOUND)
        find_path(CFITSIO_INCLUDE_DIR fitsio.h)
        find_library(CFITSIO_LIBRARY cfitsio)
        if(CFITSIO_INCLUDE_DIR AND CFITSIO_LIBRARY)
            set(CFITSIO_FOUND TRUE)
        endif()
    endif()
endif()

if(CFITSIO_FOUND)
    message(STATUS "Found environment CFITSIO: ${CFITSIO_INCLUDE_DIRS} ${CFITSIO_LIBRARIES}")
else()
    # Vendor CFITSIO as fallback
    if(NOT EXISTS "${CFITSIO_SOURCE_DIR}/CMakeLists.txt")
        torchfits_run_vendor_once()
    endif()

    if(EXISTS "${CFITSIO_SOURCE_DIR}/CMakeLists.txt")
        message(STATUS "Building vendored CFITSIO from ${CFITSIO_SOURCE_DIR}")
        
        # CFITSIO Build Options
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
        set(TESTS OFF CACHE BOOL "Disable tests" FORCE)
        set(UTILS OFF CACHE BOOL "Disable utils" FORCE)
        set(USE_PTHREADS ON CACHE BOOL "Enable thread-safe build" FORCE)
        
        # Force position-independent code for static library to be linked into shared library
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        
        add_subdirectory("${CFITSIO_SOURCE_DIR}" cfitsio_build)
        
        # CFITSIO target is named 'cfitsio'
        set(CFITSIO_LIBRARY cfitsio)
        set(CFITSIO_INCLUDE_DIR "${CFITSIO_SOURCE_DIR}")
        set(CFITSIO_FOUND TRUE)
        if(TORCHFITS_NIOBUF)
            target_compile_definitions(cfitsio PRIVATE TORCHFITS_NIOBUF=${TORCHFITS_NIOBUF})
        endif()
        if(TORCHFITS_MINDIRECT)
            target_compile_definitions(cfitsio PRIVATE TORCHFITS_MINDIRECT=${TORCHFITS_MINDIRECT})
        endif()
    else()
        message(FATAL_ERROR "CFITSIO not found in environment and vendored version not found at ${CFITSIO_SOURCE_DIR}. Run ./extern/vendor.sh")
    endif()
endif()

# Find DLPack headers (used for zero-copy tensor exchange)
find_path(DLPACK_INCLUDE_DIR dlpack/dlpack.h)
if(DLPACK_INCLUDE_DIR)
    message(STATUS "Found DLPack include: ${DLPACK_INCLUDE_DIR}")
else()
    message(WARNING "DLPack headers not found - DLPack-based zero-copy will fail to compile until headers are available")
endif()

# Find WCSLIB
option(TORCHFITS_USE_VENDORED_WCSLIB "Use vendored WCSLIB" ON)

if(NOT TORCHFITS_USE_VENDORED_WCSLIB)
    find_package(PkgConfig QUIET)
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
endif()

if(NOT WCSLIB_FOUND)
    if(NOT EXISTS "${WCSLIB_SOURCE_DIR}/CMakeLists.txt")
        torchfits_run_vendor_once()
    endif()
    if(EXISTS "${WCSLIB_SOURCE_DIR}/CMakeLists.txt")
        message(STATUS "Building vendored WCSLIB from ${WCSLIB_SOURCE_DIR}")
        add_subdirectory("${WCSLIB_SOURCE_DIR}" wcslib_build)
        set(WCSLIB_LIBRARY wcs)
        # Support both <wcs.h> and <wcslib/wcs.h> include styles.
        set(WCSLIB_INCLUDE_DIRS "${WCSLIB_SOURCE_DIR}")
        get_filename_component(WCSLIB_PARENT_DIR "${WCSLIB_SOURCE_DIR}" DIRECTORY)
        list(APPEND WCSLIB_INCLUDE_DIRS "${WCSLIB_PARENT_DIR}")
        set(WCSLIB_FOUND TRUE)
    else()
        message(FATAL_ERROR "WCSLIB not found in environment and vendored version not found at ${WCSLIB_SOURCE_DIR}. Run ./extern/vendor.sh")
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

# Avoid stale cached paths when switching environments/interpreters.
unset(TORCH_PYTHON_LIBRARY CACHE)
unset(TORCH_PYTHON_LIBRARY_FALLBACK CACHE)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH)
if(TORCH_PYTHON_LIBRARY)
    target_link_libraries(cpp PRIVATE ${TORCH_PYTHON_LIBRARY})
    message(STATUS "Found libtorch_python: ${TORCH_PYTHON_LIBRARY}")
else()
    message(WARNING "libtorch_python not found in ${TORCH_ROOT}/lib - tensor returns will use slower DLPack path")
endif()

if(TARGET cfitsio)
    target_link_libraries(cpp PRIVATE cfitsio)
    target_include_directories(cpp PRIVATE "${CFITSIO_INCLUDE_DIR}")
else()
    target_link_libraries(cpp PRIVATE ${CFITSIO_LIBRARIES} ${CFITSIO_LIBRARY})
    target_include_directories(cpp PRIVATE ${CFITSIO_INCLUDE_DIRS} ${CFITSIO_INCLUDE_DIR})
    if(CFITSIO_LIBRARY_DIRS)
        target_link_directories(cpp PRIVATE ${CFITSIO_LIBRARY_DIRS})
    endif()
endif()
target_link_libraries(cpp PRIVATE ZLIB::ZLIB)

if(DLPACK_INCLUDE_DIR)
    target_include_directories(cpp PRIVATE ${DLPACK_INCLUDE_DIR})
endif()
target_compile_definitions(cpp PRIVATE HAS_CFITSIO)

if(WCSLIB_FOUND)
    if(WCSLIB_LIBRARY_DIRS)
        target_link_directories(cpp PRIVATE ${WCSLIB_LIBRARY_DIRS})
    endif()

    if(WCSLIB_LIBRARIES)
        target_link_libraries(cpp PRIVATE ${WCSLIB_LIBRARIES})
    else()
        target_link_libraries(cpp PRIVATE ${WCSLIB_LIBRARY})
    endif()

    if(WCSLIB_INCLUDE_DIRS)
        target_include_directories(cpp PRIVATE ${WCSLIB_INCLUDE_DIRS})
        # Add parent directory of include dirs to support <wcslib/wcs.h> style includes
        foreach(DIR ${WCSLIB_INCLUDE_DIRS})
             get_filename_component(PARENT_DIR ${DIR} DIRECTORY)
             target_include_directories(cpp PRIVATE ${PARENT_DIR})
        endforeach()
    else()
        target_include_directories(cpp PRIVATE ${WCSLIB_INCLUDE_DIR})
    endif()

    target_compile_definitions(cpp PRIVATE HAS_WCSLIB)
endif()

# Compiler flags
target_compile_definitions(cpp PRIVATE VERSION_INFO="")
target_compile_features(cpp PRIVATE cxx_std_17)


# Install the compiled module
# For scikit-build-core, install to the package directory
install(TARGETS cpp DESTINATION torchfits)
