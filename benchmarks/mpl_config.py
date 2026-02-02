"""Matplotlib non-interactive defaults for benchmark runs."""

from __future__ import annotations

import os
import tempfile


def configure() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mpl-cache-")
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.usetex": False,
        }
    )
