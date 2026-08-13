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

# ---------------------------------------------------------------------------
# Profile-Guided Optimization (PGO) — opt-in via CMake option.
# Two-step build:
#   1. cmake -DTORCHFITS_PGO=generate ... && make  (run representative workload)
#   2. cmake -DTORCHFITS_PGO=use ...    && make  (rebuild with profile data)
# ---------------------------------------------------------------------------
set(TORCHFITS_PGO "off" CACHE STRING "PGO mode: off, generate, or use")
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    if(TORCHFITS_PGO STREQUAL "generate")
        add_compile_options(-fprofile-generate)
        message(STATUS "PGO: generating profile data")
    elseif(TORCHFITS_PGO STREQUAL "use")
        add_compile_options(-fprofile-use -fprofile-correction)
        message(STATUS "PGO: using profile data")
    endif()
endif()

# ---------------------------------------------------------------------------
# Finite-math-only — opt-in via CMake option.
# Assumes no NaN/Inf, enabling more aggressive vectorization and FMA fusion.
# UNSAFE for code that relies on NaN sentinel semantics (e.g., FITS null values).
# Only enable for benchmarks or when you know your codepath is NaN-free.
# ---------------------------------------------------------------------------
option(TORCHFITS_FINITE_MATH_ONLY "Assume no NaN/Inf (UNSAFE with null sentinels)" OFF)
if(TORCHFITS_FINITE_MATH_ONLY AND NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(-ffinite-math-only)
    message(STATUS "WARNING: -ffinite-math-only enabled — NaN sentinel codepaths will break")
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

message(DEBUG "torchfits: Python_EXECUTABLE=${Python_EXECUTABLE}")
message(DEBUG "torchfits: Python_VERSION=${Python_VERSION}")
message(DEBUG "torchfits: NANOBIND_PYTHON_EXECUTABLE=${NANOBIND_PYTHON_EXECUTABLE}")
message(DEBUG "torchfits: nanobind_DIR=${nanobind_DIR}")
message(DEBUG "torchfits: nanobind_ROOT=${nanobind_ROOT}")

# Find PyTorch
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__version__)"
    OUTPUT_VARIABLE TORCHFITS_BUILD_TORCH_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Require PyTorch >= 2.10; embed major.minor as the ABI tag for the runtime check.
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" _torch_mm "${TORCHFITS_BUILD_TORCH_VERSION}")
if(NOT _torch_mm)
    message(FATAL_ERROR
        "Could not parse PyTorch version '${TORCHFITS_BUILD_TORCH_VERSION}'. "
        "Install torch>=2.10 before building torchfits from source."
    )
endif()
set(TORCHFITS_TORCH_ABI_MAJOR "${CMAKE_MATCH_1}")
set(TORCHFITS_TORCH_ABI_MINOR "${CMAKE_MATCH_2}")
set(TORCHFITS_TORCH_ABI "${TORCHFITS_TORCH_ABI_MAJOR}.${TORCHFITS_TORCH_ABI_MINOR}")
if(TORCHFITS_TORCH_ABI_MAJOR LESS 2 OR (TORCHFITS_TORCH_ABI_MAJOR EQUAL 2 AND TORCHFITS_TORCH_ABI_MINOR LESS 10))
    message(FATAL_ERROR
        "torchfits requires PyTorch >= 2.10 at build time; found ${TORCHFITS_BUILD_TORCH_VERSION}. "
        "Pre-install a matching torch, then build with --no-build-isolation."
    )
endif()
message(STATUS "torchfits: building against PyTorch ${TORCHFITS_BUILD_TORCH_VERSION} (ABI ${TORCHFITS_TORCH_ABI})")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

# CUDA-enabled PyTorch only: cuda.cmake expects CUDA::nvrtc before extension targets link. Conda
# toolkits and CMake 4.x often omit that imported target; CPU-only PyTorch skips this path.
set(TORCHFITS_PYTORCH_CUDA_VERSION "")
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.version.cuda or '')"
    OUTPUT_VARIABLE TORCHFITS_PYTORCH_CUDA_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# PyTorch's cuda.cmake (pulled in by find_package(Torch)) calls get_target_property on CUDA::nvrtc
# before extension code runs. INTERFACE_LINK_LIBRARIES is read by PyTorch for nvrtc shorthash.
if(TORCHFITS_PYTORCH_CUDA_VERSION AND NOT TARGET CUDA::nvrtc)
    find_package(CUDAToolkit QUIET COMPONENTS nvrtc cudart)
endif()
if(TORCHFITS_PYTORCH_CUDA_VERSION AND NOT TARGET CUDA::nvrtc)
    set(_TORCHFITS_CUDA_LIB_HINTS "")
    foreach(_root IN ITEMS "${CUDAToolkit_ROOT}" "${CUDAToolkit_LIBRARY_ROOT}" "$ENV{CUDA_HOME}" "$ENV{CUDA_PATH}")
        if(_root)
            if(_root MATCHES "/include/?$")
                get_filename_component(_root "${_root}" DIRECTORY)
            endif()
            list(APPEND _TORCHFITS_CUDA_LIB_HINTS "${_root}/lib" "${_root}/lib64")
            list(APPEND _TORCHFITS_CUDA_LIB_HINTS "${_root}/targets/x86_64-linux/lib")
            list(APPEND _TORCHFITS_CUDA_LIB_HINTS "${_root}/targets/sbsa/lib")
        endif()
    endforeach()
    list(APPEND _TORCHFITS_CUDA_LIB_HINTS
        "$ENV{CONDA_PREFIX}/targets/x86_64-linux/lib"
        "$ENV{PREFIX}/targets/x86_64-linux/lib"
        "$ENV{CONDA_PREFIX}/lib"
        "$ENV{PREFIX}/lib"
    )
    list(REMOVE_DUPLICATES _TORCHFITS_CUDA_LIB_HINTS)
    set(TORCHFITS_NVRTC_LIBRARY "")
    foreach(_hint IN LISTS _TORCHFITS_CUDA_LIB_HINTS)
        if(NOT EXISTS "${_hint}")
            continue()
        endif()
        file(GLOB _TORCHFITS_NVRTC_GLOB "${_hint}/libnvrtc.so.*")
        if(_TORCHFITS_NVRTC_GLOB)
            list(GET _TORCHFITS_NVRTC_GLOB 0 TORCHFITS_NVRTC_LIBRARY)
            break()
        endif()
    endforeach()
    if(NOT TORCHFITS_NVRTC_LIBRARY)
        find_library(TORCHFITS_NVRTC_LIBRARY NAMES nvrtc HINTS ${_TORCHFITS_CUDA_LIB_HINTS})
    endif()
    if(NOT TORCHFITS_NVRTC_LIBRARY)
        message(STATUS "CUDA-enabled PyTorch needs libnvrtc (CUDA::nvrtc) but none was found. Creating dummy CUDA::nvrtc target.")
        add_library(CUDA::nvrtc INTERFACE IMPORTED GLOBAL)
    else()
        add_library(CUDA::nvrtc SHARED IMPORTED GLOBAL)
        set_target_properties(CUDA::nvrtc PROPERTIES
            IMPORTED_LOCATION "${TORCHFITS_NVRTC_LIBRARY}"
            INTERFACE_LINK_LIBRARIES "${TORCHFITS_NVRTC_LIBRARY}"
        )
    endif()
endif()

# Pre-create dummy `torch::cudart` (and related) interface targets so Caffe2/public/cuda.cmake's
# "poor man's include guard" at the top fires BEFORE its `find_package(CUDA)` / `find_package
# (CUDAToolkit)` / `enable_language(CUDA)` probes. This sidesteps the FATAL_ERROR
# "Your installed Caffe2 version uses CUDA but I cannot find the CUDA libraries" when pixi/conda
# envs ship only libcudart.so (no cuda.h, no nvcc). The extension is CPU-only (compile-time)
# so dummy interface targets are safe for link/load.
if(TORCHFITS_PYTORCH_CUDA_VERSION)
    foreach(_torchfits_cuda_target
            torch::cudart torch::cuda torch::nvrtc torch::cudnn torch::cufile
            caffe2::cuda caffe2::cublas caffe2::curand caffe2::cufft
            caffe2::cudnn caffe2::cufft_static caffe2::nvrtc caffe2::cudart
            caffe2::nvrtc caffe2::cudart_static caffe2::cusolver
            caffe2::cusparse caffe2::nccl caffe2::cufile caffe2::cusparselt
            caffe2::cudss caffe2::mkldnn
            CUDA::cudart CUDA::cublas CUDA::curand CUDA::cufft CUDA::cudnn
            CUDA::cusolver CUDA::cusparse CUDA::nccl CUDA::nvrtc CUDA::nvToolsExt)
        if(NOT TARGET ${_torchfits_cuda_target})
            add_library(${_torchfits_cuda_target} INTERFACE IMPORTED GLOBAL)
        endif()
    endforeach()
endif()

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
set(_TORCHFITS_ZLIB_SELECTED "")
if(DEFINED ZLIB_LIBRARY_RELEASE AND ZLIB_LIBRARY_RELEASE)
    set(_TORCHFITS_ZLIB_SELECTED "${ZLIB_LIBRARY_RELEASE}")
elseif(DEFINED ZLIB_LIBRARY AND ZLIB_LIBRARY)
    set(_TORCHFITS_ZLIB_SELECTED "${ZLIB_LIBRARY}")
endif()

if(_TORCHFITS_ZLIB_SELECTED AND NOT EXISTS "${_TORCHFITS_ZLIB_SELECTED}")
    get_filename_component(_TORCHFITS_ZLIB_DIR "${_TORCHFITS_ZLIB_SELECTED}" DIRECTORY)
    set(_TORCHFITS_ZLIB_CANDIDATES
        "${_TORCHFITS_ZLIB_DIR}/libz.1.dylib"
        "${_TORCHFITS_ZLIB_DIR}/libzlib.dylib"
    )
    foreach(_zlib_candidate IN LISTS _TORCHFITS_ZLIB_CANDIDATES)
        if(EXISTS "${_zlib_candidate}")
            message(WARNING "ZLIB library path '${_TORCHFITS_ZLIB_SELECTED}' does not exist; using '${_zlib_candidate}'")
            if(NOT EXISTS "${_TORCHFITS_ZLIB_SELECTED}")
                file(CREATE_LINK "${_zlib_candidate}" "${_TORCHFITS_ZLIB_SELECTED}" SYMBOLIC RESULT _TORCHFITS_ZLIB_LINK_RESULT)
                if(_TORCHFITS_ZLIB_LINK_RESULT)
                    message(WARNING "Failed to create compatibility symlink '${_TORCHFITS_ZLIB_SELECTED}' -> '${_zlib_candidate}': ${_TORCHFITS_ZLIB_LINK_RESULT}")
                else()
                    message(STATUS "Created compatibility symlink '${_TORCHFITS_ZLIB_SELECTED}' -> '${_zlib_candidate}'")
                endif()
            endif()
            set(ZLIB_LIBRARY_RELEASE "${_zlib_candidate}" CACHE FILEPATH "Path to a library." FORCE)
            set(ZLIB_LIBRARY "${_zlib_candidate}" CACHE FILEPATH "Path to a library." FORCE)
            if(TARGET ZLIB::ZLIB)
                set_property(TARGET ZLIB::ZLIB PROPERTY IMPORTED_LOCATION "${_zlib_candidate}")
                set_property(TARGET ZLIB::ZLIB PROPERTY IMPORTED_LOCATION_RELEASE "${_zlib_candidate}")
            endif()
            break()
        endif()
    endforeach()
endif()

if(ZLIB_FOUND)
    message(STATUS "Found ZLIB: ${ZLIB_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "ZLIB not found - required for compressed FITS support")
endif()

get_filename_component(TORCHFITS_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(CFITSIO_SOURCE_DIR "${TORCHFITS_REPO_ROOT}/extern/cfitsio")
set(TORCHFITS_VENDOR_SCRIPT "${TORCHFITS_REPO_ROOT}/extern/vendor.sh")
# Git checkouts may fetch CFITSIO on first configure. PyPI sdists (and any
# tree without .git) must not: pip installs are wheels-only.
if(EXISTS "${TORCHFITS_REPO_ROOT}/.git")
    set(_torchfits_auto_vendor_default ON)
else()
    set(_torchfits_auto_vendor_default OFF)
endif()
option(TORCHFITS_AUTO_VENDOR_DEPS "Auto-run extern/vendor.sh when vendored deps are missing" "${_torchfits_auto_vendor_default}")
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

        # CFITSIO hard-defines NIOBUF/MINDIRECT in headers. When we want to tune these
        # compile-time constants, patch in #ifndef guards once so command-line -D values
        # can actually take effect.
        if(TORCHFITS_NIOBUF OR TORCHFITS_MINDIRECT)
            set(_CFITSIO_FITSIO_H "${CFITSIO_SOURCE_DIR}/fitsio.h")
            if(EXISTS "${_CFITSIO_FITSIO_H}")
                file(READ "${_CFITSIO_FITSIO_H}" _fitsio_h_content)
                if(NOT _fitsio_h_content MATCHES "#ifndef NIOBUF")
                    string(REPLACE
                        "#define NIOBUF  40  /* number of IO buffers to create (default = 40) */"
                        "#ifndef NIOBUF\n#define NIOBUF  40  /* number of IO buffers to create (default = 40) */\n#endif"
                        _fitsio_h_content
                        "${_fitsio_h_content}"
                    )
                    file(WRITE "${_CFITSIO_FITSIO_H}" "${_fitsio_h_content}")
                endif()
            endif()

            set(_CFITSIO_FITSIO2_H "${CFITSIO_SOURCE_DIR}/fitsio2.h")
            if(EXISTS "${_CFITSIO_FITSIO2_H}")
                file(READ "${_CFITSIO_FITSIO2_H}" _fitsio2_h_content)
                if(NOT _fitsio2_h_content MATCHES "#ifndef MINDIRECT")
                    string(REPLACE
                        "#define MINDIRECT 8640   /* minimum size for direct reads and writes */"
                        "#ifndef MINDIRECT\n#define MINDIRECT 8640   /* minimum size for direct reads and writes */\n#endif"
                        _fitsio2_h_content
                        "${_fitsio2_h_content}"
                    )
                    file(WRITE "${_CFITSIO_FITSIO2_H}" "${_fitsio2_h_content}")
                endif()
            endif()
        endif()

        # CFITSIO Build Options
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
        set(TESTS OFF CACHE BOOL "Disable tests" FORCE)
        set(UTILS OFF CACHE BOOL "Disable utils" FORCE)
        set(USE_PTHREADS ON CACHE BOOL "Enable thread-safe build" FORCE)

        # Enable BZIP2_1 image compression.  Upstream CFITSIO ships it
        # disabled; the compressed image codec lives in imcompress.c and is
        # re-enabled by the vendored patch.  Requires libbz2: use the pixi /
        # conda environment when present, otherwise fall back to the system.
        option(TORCHFITS_USE_BZIP2 "Build vendored CFITSIO with BZIP2_1 support" ON)
        if(TORCHFITS_USE_BZIP2)
            if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/include/bzlib.h")
                set(USE_BZIP2 ON CACHE BOOL "Enable bzip2 support" FORCE)
                set(BZIP2_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include" CACHE FILEPATH "bzip2 include dir" FORCE)
                set(BZIP2_LIBRARY "$ENV{CONDA_PREFIX}/lib/libbz2.a" CACHE FILEPATH "bzip2 library" FORCE)
            else()
                message(STATUS "TORCHFITS_USE_BZIP2 is ON but bzlib.h was not found in the conda environment; leaving CFITSIO bzip2 support disabled")
                set(USE_BZIP2 OFF CACHE BOOL "Enable bzip2 support" FORCE)
            endif()
        endif()

        # Force position-independent code for static library to be linked into shared library
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)

        # Build and link CFITSIO, but do not install its headers, static archive,
        # pkg-config file, or CMake metadata into the Python wheel.
        add_subdirectory("${CFITSIO_SOURCE_DIR}" cfitsio_build EXCLUDE_FROM_ALL)

        # CFITSIO target is named 'cfitsio'
        set(CFITSIO_LIBRARY cfitsio)
        set(CFITSIO_INCLUDE_DIR "${CFITSIO_SOURCE_DIR}")
        set(CFITSIO_FOUND TRUE)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            # Compressed-float decode must be bit-exact across platforms.
            # Clang contracts FP expressions (FMA fusion) by default at
            # -O2/-O3, and aarch64 GCC does too, which shifts low-ULP bits
            # of unquantized / HCOMPRESS-renormalized tiles: an arm64 build
            # then reads e.g. 1.94e-16 where the lossless tile stores exact
            # 0.0, disagreeing with wheel builds that disable contraction
            # (fitsio/astropy). Pin contraction off for vendored CFITSIO.
            target_compile_options(cfitsio PRIVATE -ffp-contract=off)
        endif()
        if(TORCHFITS_NIOBUF)
            # CFITSIO reads this compile-time macro directly from fitsio.h.
            target_compile_definitions(cfitsio PRIVATE NIOBUF=${TORCHFITS_NIOBUF})
            message(STATUS "Overriding CFITSIO NIOBUF=${TORCHFITS_NIOBUF}")
        endif()
        if(TORCHFITS_MINDIRECT)
            # CFITSIO direct I/O threshold (bytes), defined in fitsio2.h.
            target_compile_definitions(cfitsio PRIVATE MINDIRECT=${TORCHFITS_MINDIRECT})
            message(STATUS "Overriding CFITSIO MINDIRECT=${TORCHFITS_MINDIRECT}")
        endif()
    else()
        message(FATAL_ERROR
            "CFITSIO sources not found at ${CFITSIO_SOURCE_DIR}. "
            "PyPI installs torchfits from prebuilt wheels (CPython 3.10-3.14, "
            "Linux x86_64/aarch64, macOS arm64) — pip should not compile. "
            "From a git checkout run: ./extern/vendor.sh --cfitsio-version extern/VERSIONS.txt")
    endif()
endif()

# Find DLPack headers (used for zero-copy tensor exchange)
find_path(DLPACK_INCLUDE_DIR dlpack/dlpack.h)
if(DLPACK_INCLUDE_DIR)
    message(STATUS "Found DLPack include: ${DLPACK_INCLUDE_DIR}")
else()
    message(WARNING "DLPack headers not found - DLPack-based zero-copy will fail to compile until headers are available")
endif()


# Source files
set(SOURCES
    bindings.cpp
    fits_bindings.cpp
    fits_file.cpp
    table_bindings.cpp
    table_ops.cpp
    cache.cpp
    hardware.cpp
)

# Create python binding module
nanobind_add_module(_C ${SOURCES})

# These optimize and isolate the Python extension. They must not propagate to
# vendored CFITSIO, whose C ABI relies on its own visibility/interposition rules.
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(_C PRIVATE -fno-semantic-interposition -fvisibility=hidden)
endif()

# Link libraries
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES})

# Link torch_python for THPVariable_Wrap (bypasses DLPack overhead)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch, os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(DEBUG "torchfits: TORCH_ROOT=${TORCH_ROOT}")

# Avoid stale cached paths when switching environments/interpreters.
unset(TORCH_PYTHON_LIBRARY CACHE)
unset(TORCH_PYTHON_LIBRARY_FALLBACK CACHE)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH)
if(TORCH_PYTHON_LIBRARY)
    target_link_libraries(_C PRIVATE ${TORCH_PYTHON_LIBRARY})
    message(STATUS "Found libtorch_python: ${TORCH_PYTHON_LIBRARY}")
else()
    # torch_compat.h calls THPVariable_Wrap unconditionally, so a missing
    # libtorch_python is a link-time failure, not a fallback to DLPack.
    message(WARNING "libtorch_python not found in ${TORCH_ROOT}/lib - torch tensor conversion may fail to link")
endif()

if(TARGET cfitsio)
    target_link_libraries(_C PRIVATE cfitsio)
    target_include_directories(_C PRIVATE "${CFITSIO_INCLUDE_DIR}")
else()
    target_link_libraries(_C PRIVATE ${CFITSIO_LIBRARIES} ${CFITSIO_LIBRARY})
    target_include_directories(_C PRIVATE ${CFITSIO_INCLUDE_DIRS} ${CFITSIO_INCLUDE_DIR})
    if(CFITSIO_LIBRARY_DIRS)
        target_link_directories(_C PRIVATE ${CFITSIO_LIBRARY_DIRS})
    endif()
endif()
target_link_libraries(_C PRIVATE ZLIB::ZLIB)

if(DLPACK_INCLUDE_DIR)
    target_include_directories(_C PRIVATE ${DLPACK_INCLUDE_DIR})
endif()
target_compile_definitions(_C PRIVATE HAS_CFITSIO)
target_compile_definitions(_C PRIVATE TORCHFITS_TORCH_ABI="${TORCHFITS_TORCH_ABI}")


# Compiler flags
target_compile_definitions(_C PRIVATE VERSION_INFO="")
target_compile_features(_C PRIVATE cxx_std_17)


# Install the compiled module
if(SKBUILD)
    # scikit-build-core (wheels): install to the package directory
    install(TARGETS _C DESTINATION torchfits)
else()
    # Bare-cmake installs (conda packages via pixi-build-cmake / rattler-build)
    # skip scikit-build-core, so mirror its wheel layout: the compiled
    # extension and the pure-Python package land in the active Python's
    # site-packages.
    set(_TORCHFITS_PYEXE "")
    if(EXISTS "${CMAKE_INSTALL_PREFIX}/bin/python")
        set(_TORCHFITS_PYEXE "${CMAKE_INSTALL_PREFIX}/bin/python")
    elseif(NANOBIND_PYTHON_EXECUTABLE)
        set(_TORCHFITS_PYEXE "${NANOBIND_PYTHON_EXECUTABLE}")
    endif()
    if(_TORCHFITS_PYEXE)
        execute_process(
            COMMAND "${_TORCHFITS_PYEXE}" -c
                    "import sysconfig; print(sysconfig.get_paths()['purelib'])"
            OUTPUT_VARIABLE _TORCHFITS_PURELIB
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(_TORCHFITS_PURELIB)
            install(TARGETS _C DESTINATION "${_TORCHFITS_PURELIB}/torchfits")
            get_filename_component(_TORCHFITS_PKG_DIR
                "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
            install(
                DIRECTORY "${_TORCHFITS_PKG_DIR}"
                DESTINATION "${_TORCHFITS_PURELIB}"
                FILES_MATCHING
                PATTERN "*.py"
                PATTERN "py.typed"
                PATTERN "cpp_src" EXCLUDE
            )
            # Console script (wheels get it from scikit-build-core entry
            # points; bare-cmake conda installs need it written out).
            get_filename_component(_TORCHFITS_BINDIR
                "${_TORCHFITS_PURELIB}/../../../bin" ABSOLUTE)
            file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/torchfits-cli"
                "#!/bin/sh\nexec \"${_TORCHFITS_BINDIR}/python\" -c 'from torchfits.cli.main import main; main()' \"\$@\"\n")
            install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/torchfits-cli"
                DESTINATION "${_TORCHFITS_BINDIR}" RENAME torchfits)
            message(STATUS "torchfits: bare-cmake install into ${_TORCHFITS_PURELIB}")
        else()
            message(WARNING "torchfits: could not locate site-packages; extension only")
        endif()
    else()
        message(WARNING "torchfits: no Python executable found; extension only")
    endif()
endif()
