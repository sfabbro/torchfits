#!/usr/bin/env python
"""Validate AI Prompting Reference Appendix in PLAN.md.
Ensures required canonical keys exist and have non-empty values.
"""
from __future__ import annotations
import re
from pathlib import Path
import sys

REQUIRED_KEYS = [
    "PACKAGE_NAME:",
    "PRIMARY_CLASSES:",
    "DATASET_CLASSES:",
    "CORE_FUNCTIONS:",
    "REMOTE_PROTOCOLS:",
    "CACHE_DIR_ENV:",
    "NAMING_CANON:",
    "PERFORMANCE_TARGETS:",
    "DOCS_STRUCTURE:",
    "TEST_CATEGORIES:",
    "BENCHMARK_OUTPUT_FMT:",
]

plan = Path("PLAN.md").read_text(encoding="utf-8")

# Find appendix fenced block
m = re.search(r"```text\n(.*?)```", plan, re.DOTALL)
if not m:
    print("AI Appendix fenced text block not found in PLAN.md", file=sys.stderr)
    sys.exit(1)
block = m.group(1)

missing = [k for k in REQUIRED_KEYS if k not in block]
if missing:
    print("Missing keys in AI appendix: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)

print("AI appendix validation passed.")
