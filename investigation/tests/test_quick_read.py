#!/usr/bin/env python3
import sys
print("Starting test...", file=sys.stderr, flush=True)

try:
    import torchfits
    print("torchfits imported", file=sys.stderr, flush=True)
    import tempfile
    from pathlib import Path
    import numpy as np
    from astropy.io import fits

    p = Path(tempfile.gettempdir()) / "test_quick.fits"
    data = np.random.randn(100, 100).astype(np.float32)
    fits.writeto(p, data, overwrite=True)
    print(f"File created: {p}", file=sys.stderr, flush=True)

    print("Testing basic read...", file=sys.stderr, flush=True)
    torchfits.clear_file_cache()
    print("Cache cleared", file=sys.stderr, flush=True)
    d, h = torchfits.read(str(p))
    print(f"SUCCESS: Read {d.shape} {d.dtype}", file=sys.stderr, flush=True)
except Exception as e:
    import traceback
    traceback.print_exc(file=sys.stderr)
    print(f"ERROR: {e}", file=sys.stderr, flush=True)
    sys.exit(1)
