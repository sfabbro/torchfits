import os

import pytest

from torchfits.openmp_guard import detect_duplicate_openmp

_omp_info = detect_duplicate_openmp()
_HAS_DUP_OMP = _omp_info.get("duplicate", False)


def pytest_configure(config):
    config._torchfits_omp_info = _omp_info  # type: ignore[attr-defined]


def pytest_report_header(config):  # pragma: no cover
    info = getattr(config, "_torchfits_omp_info", {})
    return f"torchfits OpenMP check: {info.get('message')} (candidates={info.get('candidates')})"


def pytest_collection_modifyitems(config, items):
    if _HAS_DUP_OMP:
        skip_reason = (
            "xfail due to duplicate OpenMP runtimes; environment needs consolidation"
        )
        for item in items:
            if "compressed" in item.name:
                item.add_marker(pytest.mark.xfail(reason=skip_reason, strict=False))
