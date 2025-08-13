import os
import random
import string
import tempfile

from torchfits.smart_cache import SmartCache


def test_deterministic_cache_key_and_path_small_collision_set():
    sc = SmartCache(cache_dir=None, max_size_gb=0.1)
    seen = set()
    # Use moderate N to keep test fast; larger runs can be enabled via env
    N = int(os.environ.get("TORCHFITS_CACHE_COLLISION_TEST_N", "20000"))
    for i in range(N):
        s = "".join(random.choices(string.ascii_letters + string.digits, k=32))
        key = sc._generate_key(s, hdu=None, format_type="file")
        assert key not in seen
        seen.add(key)
        # path mapping deterministic and unique per key
        rel = sc._key_to_relpath(key)
        assert rel.name == key


def test_integrity_check_and_repair_on_local_copy():
    sc = SmartCache(cache_dir=None, max_size_gb=0.1)
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src.bin")
        with open(src, "wb") as f:
            f.write(os.urandom(1024 * 32))
        # First fetch -> cached copy
        cached = sc.get_or_fetch_file(src)
        assert os.path.exists(cached)
        # Corrupt cached file
        with open(cached, "r+b") as f:
            f.seek(0)
            f.write(b"CORRUPTION")
        # Next fetch should detect mismatch and recopy from source
        cached2 = sc.get_or_fetch_file(src)
        assert cached2 == cached
        # Verify content matches source now
        with open(src, "rb") as f1, open(cached2, "rb") as f2:
            assert f1.read() == f2.read()
