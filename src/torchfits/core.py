"""
Core FITS data type support and checksum handling.
"""

from __future__ import annotations

import re


_CUTOUT_SPEC_RE = re.compile(r"(.+?)\[(\d+)\]\[(.+?)\]")


class ChecksumVerifier:
    """CFITSIO-backed FITS checksum helpers.

    FITS checksums are defined over the HDU bytes, not just an already-materialized
    ndarray. Use the file/HDU methods for exact validation.
    """

    @staticmethod
    def _validate_hdu(hdu: int) -> int:
        if isinstance(hdu, bool) or not isinstance(hdu, int):
            raise TypeError("hdu must be a non-negative integer")
        if hdu < 0:
            raise ValueError("hdu must be a non-negative integer")
        return int(hdu)

    @staticmethod
    def write_hdu_checksums(path: str, hdu: int = 0) -> None:
        """Compute and write DATASUM/CHECKSUM for one HDU."""
        import torchfits._C as cpp

        cpp.write_hdu_checksums(str(path), ChecksumVerifier._validate_hdu(hdu))

    @staticmethod
    def verify_hdu_checksums(path: str, hdu: int = 0) -> dict[str, int | bool]:
        """Verify DATASUM/CHECKSUM for one HDU using CFITSIO semantics."""
        import torchfits._C as cpp

        datastatus, hdustatus = cpp.verify_hdu_checksums(
            str(path), ChecksumVerifier._validate_hdu(hdu)
        )
        data_i = int(datastatus)
        hdu_i = int(hdustatus)
        return {
            "datastatus": data_i,
            "hdustatus": hdu_i,
            "ok": data_i == 1 and hdu_i == 1,
        }


class FITSCore:
    """Core FITS functionality."""

    @staticmethod
    def parse_cutout_spec(spec: str) -> tuple[str, int, tuple[slice, ...]]:
        """
        Parses a CFITSIO-style cutout specification string.

        Args:
            spec: The cutout specification string (e.g., 'myimage.fits[1][10:20,30:40]').

        Returns:
            A tuple containing the file path, HDU index, and a tuple of slices.
        """
        # Pattern: filename[hdu_index][slice_spec] - e.g., 'image.fits[1][10:20,30:40]'
        match = _CUTOUT_SPEC_RE.match(spec)
        if not match:
            raise ValueError(f"Invalid cutout specification: {spec}")

        file_path, hdu_str, slice_str = match.groups()
        hdu_index = int(hdu_str)

        slice_parts = slice_str.split(",")
        slices = []
        for part in slice_parts:
            try:
                if ":" in part:
                    start_str, stop_str = part.split(":")
                    start = int(start_str) if start_str else None
                    stop = int(stop_str) if stop_str else None
                    # FITS uses inclusive ranges, Python uses exclusive
                    slices.append(slice(start, stop))
                else:
                    index = int(part)
                    slices.append(slice(index, index + 1))
            except ValueError as e:
                raise ValueError(
                    f"Invalid slice specification '{part}' in cutout spec: {spec}"
                ) from e

        return file_path, hdu_index, tuple(slices)
