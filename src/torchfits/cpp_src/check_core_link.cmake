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
        # `nm -m` names a bound symbol two ways depending on the ld64 version:
        # the current single-line "(from libtorchfits_core)", and the older
        # form whose symbol line ends "referenced from:" with the library
        # named on the *next* line. Matching only "from " misreads the second
        # as unbound, so every bound CFITSIO symbol reads as unresolved and the
        # build fails with "libtorchfits_core must export CFITSIO" on a library
        # that is exporting it. Both shapes end the clause in "from:" or
        # "from ", so accept either. Guarded by
        # tests/test_cpp_self_checks.py::test_core_link_gate_accepts_both_ways_nm_names_a_bound_symbol.
        foreach(_line IN LISTS _CFITSIO_LINES)
            if(NOT _line MATCHES "from[ :]")
                list(APPEND _UNRESOLVED "${_line}")
            endif()
        endforeach()
        # Conda/pixi `nm` prints `(undefined) external _ffclos` with no
        # `(from <dylib>)` clause even when ld64 bound the symbol. That is the
        # unbound shape, so a correct build fails the gate. `/usr/bin/nm`
        # still prints the binding. Re-read with it before declaring a leak.
        if(_UNRESOLVED AND EXISTS "/usr/bin/nm" AND NOT _NM_EXECUTABLE STREQUAL "/usr/bin/nm")
            execute_process(
                COMMAND "/usr/bin/nm" -m -u "${CMAKE_TARGET}"
                OUTPUT_VARIABLE _UNDEF
                ERROR_VARIABLE _NM_ERR
                RESULT_VARIABLE _NM_RC
            )
            if(_NM_RC EQUAL 0)
                string(REGEX MATCHALL "[^\n]*_(fits|ff)[a-z0-9_]+[^\n]*" _CFITSIO_LINES "${_UNDEF}")
                set(_UNRESOLVED "")
                foreach(_line IN LISTS _CFITSIO_LINES)
                    if(NOT _line MATCHES "from[ :]")
                        list(APPEND _UNRESOLVED "${_line}")
                    endif()
                endforeach()
            endif()
        endif()
        set(_LEAKED "${_UNRESOLVED}")
    else()
        # nm could not read the target, so no symbol list exists to judge.
        # Falling through here used to set an empty result and print the
        # success message -- the gate claiming to have resolved every symbol
        # having looked at none. Mirror the ELF branch: say so and skip.
        message(STATUS "check_core_link: nm failed (${_NM_ERR}); skipping")
        return()
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
    # Undefined `ff*`/`fits_*` in a shared object is how a dynamic import
    # looks. They are filled at load when libtorchfits_core is NEEDED.
    # An empty stand-in file (the gate's own tests) has no such dependency,
    # so a genuinely unbound symbol still fails.
    if(_LEAKED)
        find_program(_READELF_EXECUTABLE NAMES readelf)
        if(_READELF_EXECUTABLE)
            execute_process(
                COMMAND "${_READELF_EXECUTABLE}" -d "${CMAKE_TARGET}"
                OUTPUT_VARIABLE _DYNAMIC
                ERROR_VARIABLE _READELF_ERR
                RESULT_VARIABLE _READELF_RC
            )
            if(_READELF_RC EQUAL 0 AND _DYNAMIC MATCHES "libtorchfits_core")
                set(_LEAKED "")
            endif()
        endif()
    endif()
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
