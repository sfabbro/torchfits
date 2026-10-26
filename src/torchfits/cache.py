"""Cache clearing, statistics, and disk-root helpers for torchfits."""

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict

CACHE_ENV_SENTINELS: tuple[str, ...] = (
    "SLURM_JOB_ID",
    "PBS_JOBID",
    "LSB_JOBID",
    "SGE_JOB_ID",
    "AWS_EXECUTION_ENV",
    "AWS_LAMBDA_FUNCTION_NAME",
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
    "AZURE_FUNCTIONS_ENVIRONMENT",
    "WEBSITE_SITE_NAME",
    "KUBERNETES_SERVICE_HOST",
    "K_SERVICE",
)


def cache_root() -> Path:
    """Filesystem root for torchfits disk caches (remote downloads, samples)."""
    override = os.environ.get("TORCHFITS_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        return Path(xdg).expanduser() / "torchfits"
    return Path.home() / ".cache" / "torchfits"


def remote_cache_root() -> Path:
    """Directory for HTTP(S) Dataset prefetch materialization."""
    override = os.environ.get("TORCHFITS_REMOTE_CACHE", "").strip()
    if override:
        return Path(override).expanduser()
    return cache_root() / "remote"


def sample_cache_root() -> Path:
    """Directory for example/gallery sample downloads."""
    override = os.environ.get("TORCHFITS_SAMPLE_CACHE", "").strip()
    if override:
        return Path(override).expanduser()
    return cache_root() / "samples"


def get_cache_stats() -> Dict[str, Any]:
    """Return in-process I/O cache statistics."""
    try:
        from ._io_engine.caches import get_cache_performance

        return get_cache_performance()
    except (ImportError, AttributeError):
        return {}


def clear_cache(*, disk: bool = False) -> None:
    """Clear in-process caches and optionally downloaded files."""
    try:
        import torchfits._C as cpp

        cpp.clear_shared_read_meta_cache()
    except (ImportError, AttributeError):
        pass
    try:
        from ._io_engine.caches import clear_file_cache

        clear_file_cache()
    except Exception as exc:
        warnings.warn(
            f"clear_cache: I/O engine clear failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    if disk:
        _remove_tree(cache_root())


def clear_all_caches() -> None:
    """Clear in-process caches and downloaded files."""
    clear_cache(disk=True)


def _remove_tree(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


__all__ = [
    "clear_all_caches",
    "clear_cache",
    "get_cache_stats",
]
