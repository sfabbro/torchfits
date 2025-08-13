#!/usr/bin/env python
"""Generate an API parity matrix vs astropy.io.fits & fitsio.

Reads a curated idioms YAML (JSON-formatted) file describing user-facing idioms,
their required symbols, and the tests that exercise them. Produces a Markdown
and JSON artifact summarizing implementation & test coverage.

Status rules:
    ✅ all required symbols found & all listed tests present
    ⚠️ partial (some symbols or tests missing)
    ❌ no symbols implemented and no tests found
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Idiom:
    id: str
    description: str
    symbols: List[str]
    tests: List[str]


def load_idioms(path: Path) -> List[Idiom]:
    if not path.exists():
        return []
    txt = path.read_text(encoding="utf-8")
    try:
        data = json.loads(txt)  # file stored as JSON for zero-dep parsing
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to parse idioms file {path}: {e}")
    idioms = []
    for entry in data.get("idioms", []):
        idioms.append(
            Idiom(
                id=entry["id"],
                description=entry.get("description", ""),
                symbols=entry.get("symbols", []),
                tests=entry.get("tests", []),
            )
        )
    return idioms


def scan_symbols(symbols: List[str]) -> Dict[str, bool]:
    found = {s: False for s in symbols}
    if not symbols:
        return found
    for py in Path("src/torchfits").rglob("*.py"):
        txt = py.read_text(encoding="utf-8", errors="ignore")
        for s in symbols:
            if not found[s] and s in txt:
                found[s] = True
        if all(found.values()):
            break
    return found


def verify_tests(tests: List[str]) -> Dict[str, bool]:
    """Each test string form: path::node::node or path::function.
    We only verify that the file exists and the final function/class substring appears in file text.
    """
    status = {}
    for t in tests:
        parts = t.split("::")
        file_path = Path(parts[0])
        leaf = parts[-1] if len(parts) > 1 else None
        ok = file_path.exists()
        if ok and leaf:
            try:
                txt = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                ok = False
            else:
                ok = (leaf in txt)
        status[t] = ok
    return status


def evaluate_idioms(idioms: List[Idiom]) -> List[Dict[str, Any]]:
    rows = []
    for idiom in idioms:
        sym_status = scan_symbols(idiom.symbols)
        test_status = verify_tests(idiom.tests)
        sym_found = all(sym_status.values()) if idiom.symbols else True
        tests_found = all(test_status.values()) if idiom.tests else True
        any_found = any(sym_status.values()) or any(test_status.values())
        if sym_found and tests_found:
            status = "✅"
        elif any_found:
            status = "⚠️"
        else:
            status = "❌"
        rows.append(
            {
                "id": idiom.id,
                "description": idiom.description,
                "symbols": idiom.symbols,
                "tests": idiom.tests,
                "symbol_status": sym_status,
                "test_status": test_status,
                "status": status,
            }
        )
    return rows


def write_outputs(rows: List[Dict[str, Any]], output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    # Markdown
    with output.open("w", encoding="utf-8") as md:
        md.write("# API Parity Matrix (Auto-generated)\n\n")
        md.write("Generated from scripts/parity/idioms.yaml (JSON).\n\n")
        md.write("| Idiom | Description | Symbols | Tests | Status |\n|-------|-------------|---------|-------|--------|\n")
        for r in rows:
            md.write(
                f"| {r['id']} | {r['description']} | {', '.join(r['symbols'])} | {len(r['tests'])} | {r['status']} |\n"
            )
        # Detail section
        md.write("\n## Details\n\n")
        for r in rows:
            md.write(f"### {r['id']} ({r['status']})\n\n")
            md.write("Symbols:\n")
            for s, ok in r["symbol_status"].items():
                md.write(f"- {s}: {'OK' if ok else 'MISSING'}\n")
            md.write("Tests:\n")
            for t, ok in r["test_status"].items():
                md.write(f"- {t}: {'OK' if ok else 'MISSING'}\n")
            md.write("\n")
    # JSON sidecar
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main():  # pragma: no cover (thin wrapper)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="Output markdown path")
    parser.add_argument(
        "--idioms", type=Path, default=Path("scripts/parity/idioms.yaml"), help="Idioms YAML (JSON) file"
    )
    args = parser.parse_args()
    idioms = load_idioms(args.idioms)
    rows = evaluate_idioms(idioms)
    write_outputs(rows, args.output)
    print(f"Parity matrix written to {args.output} ({len(rows)} idioms)")


if __name__ == "__main__":
    main()
