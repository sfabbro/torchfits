#!/usr/bin/env python
"""Generate an API parity matrix vs astropy.io.fits & fitsio (placeholder).
Scans codebase for implemented functions/classes and maps to predefined target tasks.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json

TARGETS = {
    "open_file": ["open", "FITS"],
    "list_hdus": ["get_num_hdus"],
    "read_image": ["read"],
    "read_table": ["read"],
    "header_get": ["get_header", "get_header_value"],
    "cutout": ["read"],
    "write_image": ["write_image"],
    "write_table": ["write_table"],
    "append_hdu": ["append_hdu"],
    "update_header": ["update_header"],
}

IMPLEMENTED = set()
for py in Path("src/torchfits").rglob("*.py"):
    txt = py.read_text(encoding="utf-8", errors="ignore")
    for name in {n for names in TARGETS.values() for n in names}:
        if name in txt:
            IMPLEMENTED.add(name)

rows = []
for task, names in TARGETS.items():
    implemented = all(n in IMPLEMENTED for n in names)
    rows.append({"task": task, "targets": names, "implemented": implemented})

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)

# Write markdown
with args.output.open("w", encoding="utf-8") as md:
    md.write("# API Parity Matrix (Auto-generated)\n\n")
    md.write("| Task | Required Symbols | Implemented |\n|------|------------------|-------------|\n")
    for r in rows:
        md.write(f"| {r['task']} | {', '.join(r['targets'])} | {'✅' if r['implemented'] else '❌'} |\n")

# Also JSON sidecar
args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"Parity matrix written to {args.output}")
