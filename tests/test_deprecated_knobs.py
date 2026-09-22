"""Exact deprecation texts for the removed handle-cache knobs (r4c-02/03/04).

Release 1.2 minimal-correct fallback: ``clear_file_cache(handles=)`` and
``ReadOptions.handle_cache_capacity`` are accepted-and-ignored since the handle
cache was removed; both are pinned here to their literal warning texts and
disappear in 2.0.
"""

from __future__ import annotations

import warnings

import pytest

from torchfits._io_engine.options import ReadOptions
from torchfits.io import clear_cache_subsystem, clear_file_cache

CLEAR_FILE_CACHE_HANDLES_TEXT = (
    "clear_file_cache(handles=) is ignored since the handle cache was removed; "
    "it will be removed in 2.0"
)
READ_OPTIONS_HANDLE_CACHE_CAPACITY_TEXT = (
    "ReadOptions.handle_cache_capacity is ignored since the handle cache was "
    "removed; it will be removed in 2.0"
)


def _deprecation_messages(caught):
    return [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]


def test_clear_file_cache_handles_kwarg_warns_exact_text() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clear_file_cache(handles=False)
    assert _deprecation_messages(caught) == [CLEAR_FILE_CACHE_HANDLES_TEXT]


def test_read_options_handle_cache_capacity_warns_exact_text() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ReadOptions(handle_cache_capacity=8)
    assert _deprecation_messages(caught) == [READ_OPTIONS_HANDLE_CACHE_CAPACITY_TEXT]


@pytest.mark.parametrize(
    "subsystem",
    [
        "fits_image_data",
        "fits_table_data",
        "fits_header_metadata",
        "fits_header_hdu_metadata",
        "all",
    ],
)
def test_clear_cache_subsystem_never_warns_about_handles(subsystem: str) -> None:
    """Internal policy dispatch must not surface the clear_file_cache(handles=)
    deprecation: callers of clear_cache_subsystem never passed handles=."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clear_cache_subsystem(subsystem)
    assert _deprecation_messages(caught) == []
