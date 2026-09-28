# Pure-C CFITSIO microbench against the same vendored tree as torchfits.
cmake_minimum_required(VERSION 3.15)
project(bench_cfitsio_direct C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3)
endif()

# Match torchfits cpp_src CFITSIO SIMD flags (Rice SSSE3 path on x86).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  add_compile_definitions(CFITSIO_HAVE_SSSE3)
  add_compile_options(-mssse3)
endif()

find_package(ZLIB REQUIRED)

get_filename_component(TORCHFITS_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(CFITSIO_SOURCE_DIR "${TORCHFITS_REPO_ROOT}/extern/cfitsio")
if(NOT EXISTS "${CFITSIO_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "Vendored CFITSIO missing at ${CFITSIO_SOURCE_DIR}. Run: "
    "./extern/vendor.sh --cfitsio-version extern/VERSIONS.txt")
endif()

# Match torchfits cpp_src vendored build flags (static, no utils/tests).
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(TESTS OFF CACHE BOOL "" FORCE)
set(UTILS OFF CACHE BOOL "" FORCE)
set(USE_PTHREADS ON CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory("${CFITSIO_SOURCE_DIR}" cfitsio_build EXCLUDE_FROM_ALL)

add_executable(bench_cfitsio_direct bench_cfitsio_direct.c)
target_link_libraries(bench_cfitsio_direct PRIVATE cfitsio ZLIB::ZLIB)
target_include_directories(bench_cfitsio_direct PRIVATE "${CFITSIO_SOURCE_DIR}")

install(TARGETS bench_cfitsio_direct RUNTIME DESTINATION bin)
