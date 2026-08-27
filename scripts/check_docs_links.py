#!/usr/bin/env python3
"""Crawl the built docs site and check for broken links.

Run after ``pixi run docs-build`` (or ``zensical build``) has produced
``site/``. Stdlib only — no new dependency for a CI gate.

Checks:

- Every relative ``href``/``src`` in every ``site/**/*.html`` resolves to a
  file that exists on disk (directories resolve to their ``index.html``).
- Every page has a non-empty ``<body>``.
- External ``http(s)://`` links get a best-effort HEAD/GET probe.
  First-party 404s (github.com/astroai/torchfits, astroai.github.io) fail
  the gate; other external failures are warnings only.

Exit 0 on success (no broken local links, no empty pages), exit 1 otherwise.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python < 3.11
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

SITE_DIR = Path("site")
ZENSICAL_CONFIG = Path("zensical.toml")
LINK_ATTRS = {"href", "src"}
SKIP_SCHEMES = {"mailto", "tel", "javascript", "data"}
EXTERNAL_TIMEOUT_S = 5.0


def _site_base_path() -> str:
    """Absolute hrefs are emitted for the deployed ``site_url`` path (e.g.
    ``/torchfits/...`` for a project-pages site), not the local build root.
    Strip that prefix so absolute links resolve against ``site/`` on disk."""
    if not ZENSICAL_CONFIG.is_file():
        return ""
    config = tomllib.loads(ZENSICAL_CONFIG.read_text(encoding="utf-8"))
    site_url = str(config.get("project", {}).get("site_url", ""))
    return urlsplit(site_url).path.rstrip("/")


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self._body_depth = 0
        self.body_text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "body":
            self._body_depth += 1
        attr_map = dict(attrs)
        for name in LINK_ATTRS:
            value = attr_map.get(name)
            if value:
                self.links.append(value)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "body":
            self._body_depth = max(0, self._body_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._body_depth:
            self.body_text += data


def _is_external(url: str) -> bool:
    return url.startswith(("http://", "https://", "//"))


def _resolve_local(html_file: Path, href: str, base_path: str) -> Path | None:
    """Resolve a relative/absolute local link to a filesystem path, or None
    if it should be skipped (fragment-only, empty, or unsupported scheme)."""
    scheme = urlsplit(href).scheme
    if scheme in SKIP_SCHEMES:
        return None

    path_part = urlsplit(href).path
    if not path_part:
        return None  # fragment-only (#foo) or query-only link

    if href.startswith("/"):
        if base_path and path_part.startswith(base_path):
            path_part = path_part[len(base_path) :]
        target = SITE_DIR / path_part.lstrip("/")
    else:
        target = (html_file.parent / path_part).resolve()

    return target


def _target_exists(target: Path) -> bool:
    if target.is_dir():
        return (target / "index.html").is_file()
    if target.exists():
        return True
    # mkdocs/zensical "pretty URL" style: a dirless link like "foo" may mean
    # "foo/index.html" even without a trailing slash in the href.
    return (target / "index.html").is_file()


def check_local_links(html_files: list[Path]) -> tuple[list[str], list[str]]:
    base_path = _site_base_path()
    broken: list[str] = []
    external: set[str] = set()
    for html_file in html_files:
        text = html_file.read_text(encoding="utf-8", errors="replace")
        parser = _PageParser()
        parser.feed(text)

        if not parser.body_text.strip():
            broken.append(f"{html_file}: <body> is empty")

        for href in parser.links:
            if _is_external(href):
                external.add(href)
                continue
            target = _resolve_local(html_file, href, base_path)
            if target is None:
                continue
            if not _target_exists(target):
                broken.append(f"{html_file}: broken local link {href!r} -> {target}")

    return broken, sorted(external)


def _is_first_party(url: str) -> bool:
    parts = urlsplit(url)
    host = parts.netloc.lower()
    if host in {"astroai.github.io", "www.astroai.github.io"}:
        return True
    if host in {"github.com", "www.github.com"}:
        return parts.path.startswith("/astroai/torchfits")
    return False


def _probe(url: str) -> str | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        urllib.request.urlopen(request, timeout=EXTERNAL_TIMEOUT_S)
        return None
    except urllib.error.HTTPError as exc:
        # Some hosts (e.g. unpkg.com) reject HEAD; only warn on hard 4xx/5xx
        # that also aren't a "method not allowed" false-positive.
        if exc.code in (403, 405):
            return None
        return f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - network is inherently flaky
        return str(exc)


def warn_external_links(external: list[str]) -> list[str]:
    """Probe externals. Return first-party HTTP 404s (hard failures)."""
    unique = sorted({url.split("#", 1)[0] for url in external})
    first_party_404: list[str] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for url, result in zip(unique, pool.map(_probe, unique)):
            if result is None:
                continue
            if _is_first_party(url) and result == "HTTP 404":
                first_party_404.append(f"{url} -> {result}")
                print(f"  [FAIL] {url} -> {result}")
            else:
                print(f"  [warn] {url} -> {result}")
    return first_party_404


def main() -> int:
    if not SITE_DIR.is_dir():
        print(f"[error] {SITE_DIR}/ not found. Run `pixi run docs-build` first.")
        return 1

    html_files = sorted(SITE_DIR.rglob("*.html"))
    if not html_files:
        print(f"[error] no .html files found under {SITE_DIR}/")
        return 1

    broken, external = check_local_links(html_files)

    print(f"Scanned {len(html_files)} pages, {len(external)} unique external links.")
    first_party_404: list[str] = []
    if external:
        print("Probing external links (first-party 404s fail; others warn):")
        first_party_404 = warn_external_links(external)

    if broken:
        print(f"\n[FAIL] {len(broken)} broken local link(s)/empty page(s):")
        for item in broken:
            print(f"  - {item}")
    if first_party_404:
        print(f"\n[FAIL] {len(first_party_404)} first-party 404(s):")
        for item in first_party_404:
            print(f"  - {item}")
    if broken or first_party_404:
        return 1

    print("[OK] no broken local links, no empty pages, no first-party 404s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
