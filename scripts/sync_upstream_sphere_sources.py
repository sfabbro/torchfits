#!/usr/bin/env python3
"""Sync upstream sphere package source artifacts and extract tests/data fixtures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACKAGE_ALIASES: dict[str, str] = {
    "healpy": "healpy",
    "astropy-healpix": "astropy-healpix",
    "hpgeom": "hpgeom",
    "healpix": "healpix",
    "mhealpy": "mhealpy",
    "healsparse": "healsparse",
    "skyproj": "skyproj",
    "spherical-geometry": "spherical_geometry",
}

TEST_DIR_NAMES = {"test", "tests", "testing"}
DATA_DIR_NAMES = {"data", "test_data", "tests_data"}


@dataclass(frozen=True)
class DistMeta:
    name: str
    version: str
    release_like: bool
    source: str
    installer: str
    reason: str


def _installed_package_roots(dist_name: str) -> list[Path]:
    dist = importlib_metadata.distribution(dist_name)
    root = Path(dist.locate_file(""))
    top_raw = dist.read_text("top_level.txt") or ""
    top_levels = [x.strip() for x in top_raw.splitlines() if x.strip()]
    if not top_levels:
        top_levels = [dist_name.replace("-", "_")]
    roots: list[Path] = []
    for top in top_levels:
        p = root / top
        if p.exists():
            roots.append(p)
    return roots


def _distribution_meta(dist_name: str) -> DistMeta | None:
    try:
        dist = importlib_metadata.distribution(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return None

    installer = (dist.read_text("INSTALLER") or "unknown").strip().lower() or "unknown"
    source = "site-packages"
    release_like = True
    reason = "installed from index/conda/wheel/sdist"

    if installer == "conda":
        reason = "conda package"
        return DistMeta(
            name=dist.metadata.get("Name", dist_name),
            version=dist.version,
            release_like=release_like,
            source=source,
            installer=installer,
            reason=reason,
        )

    raw_direct = dist.read_text("direct_url.json")
    if raw_direct:
        try:
            direct = json.loads(raw_direct)
        except json.JSONDecodeError:
            direct = {}
        url = str(direct.get("url", "")).strip()
        if url:
            source = url
        if direct.get("vcs_info") is not None:
            release_like = False
            reason = "VCS install"
        elif direct.get("dir_info") is not None and direct.get("archive_info") is None:
            release_like = False
            reason = "local path/editable install"

    return DistMeta(
        name=dist.metadata.get("Name", dist_name),
        version=dist.version,
        release_like=release_like,
        source=source,
        installer=installer,
        reason=reason,
    )


def _parse_aliases(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    vals = {x.strip().lower() for x in raw.split(",") if x.strip()}
    return vals or None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_archive(artifact: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    name = artifact.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar")):
        with tarfile.open(artifact, "r:*") as tf:
            tf.extractall(dst)
        return
    if name.endswith((".whl", ".zip")):
        with zipfile.ZipFile(artifact, "r") as zf:
            zf.extractall(dst)
        return
    raise ValueError(f"Unsupported artifact format: {artifact}")


def _find_fixture_dirs(root: Path) -> dict[str, list[str]]:
    tests: list[str] = []
    data: list[str] = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        n = p.name.lower()
        rel = str(p.relative_to(root))
        if n in TEST_DIR_NAMES:
            tests.append(rel)
        if n in DATA_DIR_NAMES:
            data.append(rel)
    tests.sort()
    data.sort()
    return {"tests_dirs": tests, "data_dirs": data}


def _snapshot_from_installed(dist_name: str, dst_root: Path) -> None:
    roots = _installed_package_roots(dist_name)
    if not roots:
        raise RuntimeError("no importable package roots found in installed distribution")
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    for src in roots:
        dst = dst_root / src.name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _download_artifact(
    dist_name: str,
    version: str,
    download_dir: Path,
    prefer_sdist: bool,
) -> Path:
    before = {p.name for p in download_dir.glob("*")}
    base_cmd = [sys.executable, "-m", "pip", "download", "--no-deps", "--dest", str(download_dir)]
    spec = f"{dist_name}=={version}"

    attempted: list[list[str]] = []
    if prefer_sdist:
        attempted.append(base_cmd + ["--no-binary", ":all:", spec])
    attempted.append(base_cmd + [spec])

    last_err: str | None = None
    for cmd in attempted:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if proc.returncode == 0:
            after = {p.name for p in download_dir.glob("*")}
            new_files = sorted((after - before))
            if not new_files:
                # Might already exist from previous run; use newest matching spec.
                candidates = sorted(download_dir.glob(f"{dist_name.replace('-', '_')}*"), key=lambda p: p.stat().st_mtime)
                if candidates:
                    return candidates[-1]
                candidates = sorted(download_dir.glob("*"), key=lambda p: p.stat().st_mtime)
                if candidates:
                    return candidates[-1]
                last_err = "download succeeded but no artifact found"
                continue
            candidates = [download_dir / n for n in new_files]
            candidates.sort(key=lambda p: p.stat().st_mtime)
            return candidates[-1]
        stderr = (proc.stderr or "").strip().splitlines()
        stdout = (proc.stdout or "").strip().splitlines()
        tail = stderr[-1] if stderr else (stdout[-1] if stdout else "unknown pip download error")
        last_err = f"cmd={' '.join(cmd)} :: {tail}"

    raise RuntimeError(last_err or f"Failed to download {dist_name}=={version}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/upstream_fixtures"),
        help="Root directory for artifacts, extracted sources, and manifest",
    )
    parser.add_argument("--include", type=str, default=None, help="Comma-separated package aliases to include")
    parser.add_argument("--exclude", type=str, default=None, help="Comma-separated package aliases to exclude")
    parser.add_argument("--prefer-sdist", action="store_true", help="Try source distributions first")
    parser.add_argument(
        "--no-installed-fallback",
        action="store_true",
        help="Disable fallback snapshot from installed site-packages when download fails",
    )
    parser.add_argument(
        "--allow-nonrelease-installed",
        action="store_true",
        help="Allow syncing versions from local/editable/VCS installed distributions",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on missing/failed packages")
    args = parser.parse_args()

    include = _parse_aliases(args.include)
    exclude = _parse_aliases(args.exclude) or set()

    out = args.output_dir
    downloads = out / "downloads"
    sources = out / "sources"
    out.mkdir(parents=True, exist_ok=True)
    downloads.mkdir(parents=True, exist_ok=True)
    sources.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"packages": []}
    failures: list[str] = []

    for alias, dist_name in PACKAGE_ALIASES.items():
        a = alias.lower()
        if include is not None and a not in include:
            continue
        if a in exclude:
            continue

        meta = _distribution_meta(dist_name)
        if meta is None:
            failures.append(f"{alias}: not installed in current environment")
            continue
        if not args.allow_nonrelease_installed and not meta.release_like:
            failures.append(f"{alias}: installed distribution is non-release ({meta.reason}: {meta.source})")
            continue

        print(f"[sync] {alias}=={meta.version}")
        extract_root = sources / f"{alias}-{meta.version}"
        try:
            artifact = _download_artifact(
                dist_name=dist_name,
                version=meta.version,
                download_dir=downloads,
                prefer_sdist=args.prefer_sdist,
            )
            sha = _sha256(artifact)
            _extract_archive(artifact, extract_root)
            fixture_dirs = _find_fixture_dirs(extract_root)
            record = {
                "alias": alias,
                "dist_name": dist_name,
                "version": meta.version,
                "installed_meta": {
                    "release_like": meta.release_like,
                    "source": meta.source,
                    "installer": meta.installer,
                    "reason": meta.reason,
                },
                "artifact": {
                    "path": str(artifact),
                    "filename": artifact.name,
                    "sha256": sha,
                    "origin": "download",
                },
                "extracted_root": str(extract_root),
                "tests_dirs": fixture_dirs["tests_dirs"],
                "data_dirs": fixture_dirs["data_dirs"],
            }
            manifest["packages"].append(record)
        except Exception as exc:  # pragma: no cover - operational failures
            if args.no_installed_fallback:
                failures.append(f"{alias}: {exc}")
                continue
            try:
                _snapshot_from_installed(dist_name=dist_name, dst_root=extract_root)
                fixture_dirs = _find_fixture_dirs(extract_root)
                record = {
                    "alias": alias,
                    "dist_name": dist_name,
                    "version": meta.version,
                    "installed_meta": {
                        "release_like": meta.release_like,
                        "source": meta.source,
                        "installer": meta.installer,
                        "reason": meta.reason,
                    },
                    "artifact": {
                        "path": "site-packages snapshot",
                        "filename": "",
                        "sha256": "",
                        "origin": "installed-fallback",
                        "download_error": str(exc),
                    },
                    "extracted_root": str(extract_root),
                    "tests_dirs": fixture_dirs["tests_dirs"],
                    "data_dirs": fixture_dirs["data_dirs"],
                }
                manifest["packages"].append(record)
            except Exception as fallback_exc:
                failures.append(f"{alias}: {exc} | fallback failed: {fallback_exc}")

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest written: {manifest_path}")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
