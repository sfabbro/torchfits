#!/usr/bin/env python3
"""Run all benchmark scenarios with sensible defaults."""
from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]):
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    py = sys.executable
    run([py, "benchmarks/table_bulk_read_micro.py"])  # prints comparison table
    run([py, "benchmarks/image_cutout_compare.py", "--size", "1024", "--cutouts", "10", "--cutout-size", "64", "--reps", "5"])  # cutouts
    run([py, "benchmarks/compare_readers.py", "--size", "1024", "--cutouts", "10", "--cutout-size", "64", "--reps", "5",
         "--mef-hdus", "3", "--files", "3", "--sky-cutouts", "10", "--sky-radius-arcsec", "30"])  # all-in-one
    run([py, "benchmarks/cutout_batch_micro.py", "--batch", "64", "--rep", "5"])  # batched cutouts


if __name__ == "__main__":
    main()
