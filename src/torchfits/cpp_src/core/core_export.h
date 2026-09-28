#pragma once

// Symbol visibility for the torch-free core shared library.
//
// The core is built without -fvisibility=hidden (CFITSIO and its helpers have
// their own interposition rules that must not be disturbed), so this macro is
// a no-op on the platforms torchfits ships wheels for. It exists so the same
// sources still link on Windows, where a DLL only exports what it marks.
#if defined(_WIN32)
#  if defined(TORCHFITS_CORE_BUILDING)
#    define TORCHFITS_CORE_API __declspec(dllexport)
#  else
#    define TORCHFITS_CORE_API __declspec(dllimport)
#  endif
#else
#  define TORCHFITS_CORE_API __attribute__((visibility("default")))
#endif
