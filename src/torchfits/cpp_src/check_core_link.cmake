# Post-link guard for the torch-free core split.
#
# libtorchfits_core owns the one and only copy of CFITSIO. The torch-linked
# extension (_C) calls fits_* directly, so its link line resolves those symbols
# against the core's dynamic symbol table. If that ever stops happening -- a
# stale core, a stripped export, an LTO pass internalizing a symbol -- _C
# would still build (undefined symbols in a shared object are legal) and then
# fail at import time with a bare "undefined symbol: _ffopen". Fail the build
# here instead, naming the symbol.
#
# Invoked as: cmake -DCMAKE_TARGET=<lib> -P check_core_link.cmake
if(NOT CMAKE_TARGET)
    message(FATAL_ERROR "check_core_link.cmake requires -DCMAKE_TARGET=<library>")
endif()

find_program(_NM_EXECUTABLE NAMES nm)
if(NOT _NM_EXECUTABLE)
    message(STATUS "check_core_link: nm not found; skipping undefined-CFITSIO check")
    return()
endif()

if(APPLE)
    # macOS uses a two-level namespace: `nm -u` lists every symbol _C does not
    # define itself, including the ones the linker bound to a specific dylib.
    # `nm -m` appends "from <dylib>" to those, so a CFITSIO symbol with no
    # "from" clause is the only genuinely unresolved case -- and it is also the
    # only one that would fail at import time.
    execute_process(
        COMMAND "${_NM_EXECUTABLE}" -m -u "${CMAKE_TARGET}"
        OUTPUT_VARIABLE _UNDEF
        ERROR_VARIABLE _NM_ERR
        RESULT_VARIABLE _NM_RC
    )
    if(_NM_RC EQUAL 0)
        string(REGEX MATCHALL "[^\n]*_(fits|ff)[a-z0-9_]+[^\n]*" _CFITSIO_LINES "${_UNDEF}")
        set(_UNRESOLVED "")
        foreach(_line IN LISTS _CFITSIO_LINES)
            if(NOT _line MATCHES "from ")
                list(APPEND _UNRESOLVED "${_line}")
            endif()
        endforeach()
        set(_LEAKED "${_UNRESOLVED}")
    else()
        set(_LEAKED "")
    endif()
else()
    # ELF: a shared object may legally carry undefined symbols and let the
    # dynamic linker fill them in, so check the dynamic table directly.
    execute_process(
        COMMAND "${_NM_EXECUTABLE}" -D --undefined-only "${CMAKE_TARGET}"
        OUTPUT_VARIABLE _UNDEF
        ERROR_VARIABLE _NM_ERR
        RESULT_VARIABLE _NM_RC
    )
    if(NOT _NM_RC EQUAL 0)
        message(STATUS "check_core_link: nm failed (${_NM_ERR}); skipping")
        return()
    endif()
    string(REGEX MATCHALL "[ \t_](fits|ff)[a-z0-9_]+" _LEAKED "${_UNDEF}")
endif()

if(_LEAKED)
    list(REMOVE_DUPLICATES _LEAKED)
    list(LENGTH _LEAKED _N)
    list(SUBLIST _LEAKED 0 12 _HEAD)
    string(REPLACE ";" ", " _HEAD_STR "${_HEAD}")
    if(_N GREATER 12)
        set(_SUFFIX ", ...")
    else()
        set(_SUFFIX "")
    endif()
    message(FATAL_ERROR
        "${CMAKE_TARGET} has ${_N} undefined CFITSIO symbol(s): ${_HEAD_STR}${_SUFFIX}\n"
        "libtorchfits_core must export CFITSIO for _C to bind against. Check that\n"
        "the core target is on _C's link line and is not built with -fvisibility=hidden.")
endif()
message(STATUS "check_core_link: ${CMAKE_TARGET} resolves every CFITSIO symbol")
