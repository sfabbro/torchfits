#!/usr/bin/env python
"""Naming consistency gate for torchfits.

Rules enforced:
- Package import must be `import torchfits` (no CamelCase variant) — checked indirectly by scanning for `TorchFits` token.
- Disallow banned identifiers: TorchFits, Fits, FitsTable (temporary until alias deprecation strategy implemented).
- Allow specific uppercase classes: FITS, HDU, WCS.
- Function and module names should be snake_case (basic heuristic by flagging CamelCase function definitions).

Exit code 1 if violations found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Tokens explicitly forbidden (case sensitive)
BANNED = {"TorchFits", "FitsTable", "FitsDataset", "Fits"}
ALLOWED_CLASS_UPPER = {"FITS", "HDU", "WCS"}

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "torchfits"

camel_func_pattern = re.compile(r"^def\s+([A-Z][A-Za-z0-9]+)\(")
class_name_pattern = re.compile(r"^class\s+([A-Za-z0-9_]+)\(")

violations: list[str] = []

for py in SRC.rglob("*.py"):
    if py.name == "__init__.py":
        continue
    rel = py.relative_to(ROOT)
    text = py.read_text(encoding="utf-8", errors="ignore")
    # Token bans
    for token in BANNED:
        if token in text:
            violations.append(f"{rel}: banned token '{token}'")
    # Function defs CamelCase
    for m in re.finditer(r"^def\s+([A-Z][A-Za-z0-9_]+)\(", text, re.MULTILINE):
        fname = m.group(1)
        violations.append(f"{rel}: function name '{fname}' should be snake_case")
    # Classes not following policy (allow uppercase canonical, CamelCase otherwise ok)
    for m in re.finditer(r"^class\s+([A-Za-z0-9_]+)\(", text, re.MULTILINE):
        cname = m.group(1)
        if cname.isupper() and cname not in ALLOWED_CLASS_UPPER:
            violations.append(f"{rel}: all-caps class '{cname}' not allowed (allowed: {ALLOWED_CLASS_UPPER})")
        if cname == "FitsTable":  # transitional alias detection
            violations.append(f"{rel}: class 'FitsTable' should be renamed (keep alias with deprecation)")

if violations:
    print("Naming policy violations detected:\n" + "\n".join(violations))
    sys.exit(1)

print("Naming policy check passed.")
