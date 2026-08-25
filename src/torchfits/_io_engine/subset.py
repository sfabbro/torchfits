"""Image subset/cutout readers for FITS I/O."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Type, cast

from torch import Tensor

from .device import to_device, validate_device
from .http_subset import HttpRangeUnsupported, read_subset_http
from .paths import require_bz2_support
from torchfits.http_util import HttpRangeNotSatisfied


def _remote_helpers() -> tuple[
    Callable[[str], bool],
    Callable[[str], bool],
    Callable[..., str],
]:
    from torchfits.data.remote import is_http_url, is_vos_path, resolve_local_path

    return is_http_url, is_vos_path, resolve_local_path


def read_subset(
    get_cached_handle: Callable[[str, int], tuple[Any, bool]],
    path: str,
    hdu: int | str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    handle_cache_capacity: int = 16,
) -> Tensor:
    """Read a rectangular subset of an image HDU."""
    import torchfits._C as cpp
    from .paths import guard_fits_path

    guard_fits_path(path)
    is_http_url, is_vos_path, resolve_local_path = _remote_helpers()
    if is_http_url(path):
        try:
            return read_subset_http(path, hdu, x1, y1, x2, y2)
        except (HttpRangeUnsupported, HttpRangeNotSatisfied):
            path = resolve_local_path(path)
    elif is_vos_path(path):
        path = resolve_local_path(path)

    if isinstance(hdu, str):
        if hasattr(cpp, "resolve_hdu_name_cached"):
            hdu = int(cpp.resolve_hdu_name_cached(path, hdu))
        else:
            raise ValueError("named HDUs require resolve_hdu_name_cached support")
    try:
        file_handle, cached = get_cached_handle(path, handle_cache_capacity)
        try:
            return cast(Tensor, file_handle.read_subset(hdu, x1, y1, x2, y2))
        finally:
            if not cached:
                try:
                    file_handle.close()
                except Exception:
                    pass
    except Exception as exc:
        raise RuntimeError(f"Failed to read subset from '{path}': {exc}") from exc


class SubsetReader:
    """Persistent subset reader for repeated cutouts on one image HDU."""

    def __init__(self, path: str, hdu: int | str = 0, device: str = "cpu"):
        import torchfits._C as cpp
        from .paths import guard_fits_path

        if not isinstance(path, str):
            raise ValueError("path must be a string")
        guard_fits_path(path)
        require_bz2_support(path)
        if not isinstance(hdu, (int, str)):
            raise ValueError("hdu must be an integer or string")
        if isinstance(hdu, int) and hdu < 0:
            raise ValueError("hdu must be a non-negative integer")
        validate_device(device)
        self._http_url: str | None = None
        self._http_hdu: int | str = hdu
        self._device = device
        self._reader: Any = None
        self._shape: Tuple[int, int] | None = None

        is_http_url, is_vos_path, resolve_local_path = _remote_helpers()
        if is_http_url(path):
            try:
                from .http_subset import locate_uncompressed_2d

                meta = locate_uncompressed_2d(path, hdu)
                self._http_url = path
                self._shape = (int(meta["naxis2"]), int(meta["naxis1"]))
                return
            except (HttpRangeUnsupported, HttpRangeNotSatisfied):
                path = resolve_local_path(path)
        elif is_vos_path(path):
            path = resolve_local_path(path)

        if isinstance(hdu, str):
            if hasattr(cpp, "resolve_hdu_name_cached"):
                hdu = int(cpp.resolve_hdu_name_cached(path, hdu))
            else:
                raise ValueError("named HDUs require resolve_hdu_name_cached support")

        self._reader = cpp.SubsetReader(path, int(hdu))

    @property
    def hdu(self) -> int:
        if self._http_url is not None:
            return int(self._http_hdu) if isinstance(self._http_hdu, int) else -1
        return int(self._reader.hdu)

    @property
    def shape(self) -> Tuple[int, int]:
        if self._shape is not None:
            return self._shape
        return int(self._reader.height), int(self._reader.width)

    def read_subset(self, x1: int, y1: int, x2: int, y2: int) -> Tensor:
        if self._http_url is not None:
            try:
                out: Tensor = read_subset_http(
                    self._http_url, self._http_hdu, x1, y1, x2, y2
                )
            except (HttpRangeUnsupported, HttpRangeNotSatisfied):
                import torchfits._C as cpp

                _, _, resolve_local_path = _remote_helpers()
                path = resolve_local_path(self._http_url)
                hdu = self._http_hdu
                if isinstance(hdu, str):
                    hdu = int(cpp.resolve_hdu_name_cached(path, hdu))
                self._http_url = None
                self._reader = cpp.SubsetReader(path, int(hdu))
                self._shape = None
                out = cast(
                    Tensor, self._reader.read(int(x1), int(y1), int(x2), int(y2))
                )
        else:
            out = cast(Tensor, self._reader.read(int(x1), int(y1), int(x2), int(y2)))
        if str(self._device) != "cpu":
            out = to_device(out, self._device)
        return out

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()

    def __call__(self, x1: int, y1: int, x2: int, y2: int) -> Tensor:
        return self.read_subset(x1, y1, x2, y2)

    def __enter__(self) -> SubsetReader:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Any,
    ) -> None:
        self.close()


def open_subset_reader(
    path: str, hdu: int | str = 0, device: str = "cpu"
) -> SubsetReader:
    """Open a persistent cutout reader for repeated subsets on one HDU."""
    return SubsetReader(path=path, hdu=hdu, device=device)
