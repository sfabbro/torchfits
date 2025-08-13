"""
Example: Remote reading with SmartCache and integrity-preserving local cache.

Demonstrates reading a remote FITS over HTTPS, auto-caching to TORCHFITS_CACHE,
then reading again to show a cache hit. Also shows how to configure the cache
location and print cache statistics.
"""

import os
import sys

import torchfits as tf


def main():
    # Optional: direct the cache to a specific base path
    os.environ.setdefault(
        "TORCHFITS_CACHE", os.path.expanduser("~/.cache/torchfits-demo")
    )

    # Configure cache (optional; defaults are sensible)
    tf.configure_cache(max_size_gb=1.0, max_files=200)

    # Provide a remote URL via env to avoid hardcoding and allow offline runs.
    # Example: export TORCHFITS_REMOTE_URL="https://example.org/data/sample.fits"
    url = os.environ.get("TORCHFITS_REMOTE_URL")
    if not url:
        print("Set TORCHFITS_REMOTE_URL to a valid http(s) FITS to run this demo.")
        print(
            'Example: export TORCHFITS_REMOTE_URL="https://example.org/data/sample.fits"'
        )
        sys.exit(0)

    # First read will fetch and cache; second should be faster
    print("Reading remote FITS (first time; expect cache MISS)...")
    data1, hdr1 = tf.read(url)
    print(f"Primary shape: {getattr(data1, 'shape', None)}")

    print("Reading remote FITS (second time; expect cache HIT)...")
    data2, hdr2 = tf.read(url)
    print(f"Primary shape: {getattr(data2, 'shape', None)}")

    # Print cache statistics
    stats = tf.get_cache_manager().get_cache_statistics()
    print("Cache stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
