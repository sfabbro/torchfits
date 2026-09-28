"""Download and cache public FITS samples for gallery examples."""

from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


def _default_sample_cache() -> Path:
    override = os.environ.get("TORCHFITS_SAMPLE_CACHE", "").strip()
    if override:
        return Path(override).expanduser()
    try:
        from torchfits.cache import sample_cache_root

        return sample_cache_root()
    except Exception:
        xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
        if xdg:
            return Path(xdg).expanduser() / "torchfits" / "samples"
        return Path.home() / ".cache" / "torchfits" / "samples"


CACHE_DIR = _default_sample_cache()

# Stable public tutorial / survey files (astropy-data + SDSS SAS).
SAMPLES: dict[str, str] = {
    "horsehead": "http://data.astropy.org/tutorials/FITS-images/HorseHead.fits",
    "chandra_events": "http://data.astropy.org/tutorials/FITS-tables/chandra_events.fits",
    # Same plate/mjd/fiber used in specutils Spectrum.read docs.
    "sdss_spectrum": (
        "https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/"
        "0751/spec-0751-52251-0160.fits"
    ),
    "m13_blue_0001": "http://data.astropy.org/tutorials/FITS-images/M13_blue_0001.fits",
    "m13_blue_0002": "http://data.astropy.org/tutorials/FITS-images/M13_blue_0002.fits",
    "m13_blue_0003": "http://data.astropy.org/tutorials/FITS-images/M13_blue_0003.fits",
    "m13_blue_0004": "http://data.astropy.org/tutorials/FITS-images/M13_blue_0004.fits",
    "m13_blue_0005": "http://data.astropy.org/tutorials/FITS-images/M13_blue_0005.fits",
    "fits_header_mef": "http://data.astropy.org/tutorials/FITS-Header/input_file.fits",
    "sdss_lupton_g": "http://data.astropy.org/visualization/reprojected_sdss_g.fits.bz2",
    "sdss_lupton_r": "http://data.astropy.org/visualization/reprojected_sdss_r.fits.bz2",
    "sdss_lupton_i": "http://data.astropy.org/visualization/reprojected_sdss_i.fits.bz2",
    "spitzer_example": "http://data.astropy.org/photometry/spitzer_example_image.fits",
    "radio_cube_c14": "http://data.astropy.org/tutorials/FITS-cubes/reduced_TAN_C14.fits",
    "manga_logcube": (
        "https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/7443/"
        "stack/manga-7443-12703-LOGCUBE.fits.gz"
    ),
    "galaxy_zoo1_table2": (
        "https://galaxy-zoo-1.s3.amazonaws.com/GalaxyZoo1_DR_table2.fits"
    ),
}


class SampleUnavailable(RuntimeError):
    """Raised when network samples cannot be fetched (or FAST mode skips)."""


# Per-sample socket timeout for the fetch. urlopen applies it to each socket
# operation, not to the whole transfer, so the ~200 MB manga_logcube still
# downloads while a black-holed host fails here instead of blocking until the
# OS gives up on the TCP handshake.
SAMPLE_TIMEOUT_S = 30.0

# No valid FITS file is smaller than one 2880-byte block, so anything below
# that is a partial download rather than a sample. Every entry in SAMPLES is
# far above it; the smallest one, fits_header_mef.fits, is 218,880 bytes.
_MIN_SAMPLE_BYTES = 2880

# _dest_path() keeps the URL's (possibly compound) suffix, so a cached sample
# is plain FITS, bzip2 or gzip. Each container announces itself in its first
# few bytes; an HTML error page, a proxy notice or a torn transfer does not.
_MAGIC_BY_SUFFIX: dict[str, tuple[bytes, ...]] = {
    ".fits": (b"SIMPLE  =",),
    ".fits.bz2": (b"BZh",),
    ".fits.gz": (b"\x1f\x8b",),
}


def megacam_dir() -> Path:
    """Local cache dir for CFHT MegaCam ``.fits.fz`` samples (see fetch script)."""
    return Path(__file__).resolve().parents[1] / "benchmarks_data" / "cfht_megacam"


def megapipe_dir() -> Path:
    """Local cache dir for CFHTLS-Deep D1 MegaPipe mosaics/catalog (see fetch script)."""
    return Path(__file__).resolve().parents[1] / "benchmarks_data" / "cfht_megapipe"


def gz_legacy_cutouts_dir() -> Path:
    """Cache dir for Legacy Survey grz cutouts keyed to Galaxy Zoo 1 rows."""
    return CACHE_DIR / "gz_legacy_cutouts"


def _url_suffix(name: str) -> str:
    """The (possibly compound) suffix of the sample's URL, e.g. ``.fits.bz2``."""
    url_name = Path(urlsplit(SAMPLES[name]).path).name
    return "".join(Path(url_name).suffixes) or ".fits"


def _dest_path(name: str) -> Path:
    """Cache path for ``name``, preserving the URL's (possibly compound) suffix."""
    return CACHE_DIR / f"{name}{_url_suffix(name)}"


def _is_sample_file(name: str, path: Path) -> bool:
    """True when ``path`` is a plausible copy of the sample it stands for.

    A cache hit used to be accepted on ``st_size > 0`` alone, so a truncated
    or garbage entry was served to every later run: each example that opened
    it failed with "Could not open FITS file", and because the file was in
    the cache nothing ever re-fetched it. The size floor alone would miss a
    mid-file truncation of a large sample, so the container magic is checked
    as well.
    """
    magics = _MAGIC_BY_SUFFIX.get(_url_suffix(name))
    if magics is None:
        # An unrecognised container: the size floor is all we can check.
        return path.is_file() and path.stat().st_size >= _MIN_SAMPLE_BYTES
    try:
        if path.stat().st_size < _MIN_SAMPLE_BYTES:
            return False
        with path.open("rb") as fh:
            head = fh.read(max(len(m) for m in magics))
    except OSError:
        return False
    return head in magics


def _fast_mode() -> bool:
    return os.environ.get("TORCHFITS_EXAMPLE_FAST", "").strip() in (
        "1",
        "true",
        "TRUE",
        "yes",
    )


def ensure_sample(name: str, *, allow_download: bool | None = None) -> Path:
    """Return a local path for a named sample, downloading once if needed.

    In ``TORCHFITS_EXAMPLE_FAST=1`` (CI), skips network and raises
    :class:`SampleUnavailable` unless the file is already cached.
    """
    if name not in SAMPLES:
        raise KeyError(f"unknown sample {name!r}; choose from {sorted(SAMPLES)}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _dest_path(name)
    evicted = ""
    if dest.is_file():
        if _is_sample_file(name, dest):
            return dest
        size = dest.stat().st_size
        dest.unlink(missing_ok=True)
        evicted = f"; removed an unusable {size}-byte cached copy"

    if allow_download is None:
        allow_download = not _fast_mode()
    if not allow_download:
        raise SampleUnavailable(
            f"sample {name!r} not usable at {dest}{evicted} "
            f"(TORCHFITS_EXAMPLE_FAST skips download)"
        )

    url = SAMPLES[name]
    tmp = dest.with_name(dest.name + ".partial")
    try:
        # urlopen, not urlretrieve: urlretrieve takes no timeout and blocks
        # until the OS gives up on the connection. copyfileobj keeps the
        # transfer streamed to disk so the ~200 MB manga_logcube is never held
        # in memory in one piece.
        with urllib.request.urlopen(url, timeout=SAMPLE_TIMEOUT_S) as response:  # noqa: S310 — fixed public URLs
            with tmp.open("wb") as fh:
                shutil.copyfileobj(response, fh)
        if not _is_sample_file(name, tmp):
            raise OSError(
                f"transfer ended with {tmp.stat().st_size} bytes that are not a "
                f"{name} sample"
            )
        tmp.replace(dest)
    except (urllib.error.URLError, OSError) as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise SampleUnavailable(f"failed to download {name} from {url}: {exc}") from exc
    return dest


def try_ensure_sample(name: str) -> Path | None:
    """Like :func:`ensure_sample` but returns ``None`` when unavailable."""
    try:
        return ensure_sample(name)
    except SampleUnavailable:
        return None
