from __future__ import annotations

import ast
from pathlib import Path


def test_no_astropy_or_fitsio_imports_in_torchfits_python_package() -> None:
    pkg_root = Path(__file__).resolve().parents[1] / "src" / "torchfits"
    forbidden = {"astropy", "fitsio"}
    violations: list[str] = []

    for py_file in pkg_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".", 1)[0]
                    if top in forbidden:
                        violations.append(f"{py_file}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                top = node.module.split(".", 1)[0]
                if top in forbidden:
                    violations.append(f"{py_file}: from {node.module} import ...")

    assert not violations, "Forbidden backend imports found:\n" + "\n".join(violations)
