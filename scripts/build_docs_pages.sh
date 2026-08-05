#!/usr/bin/env bash
# Build GitHub Pages tree with stable (/) + edge (/edge/) docs.
#
#   site_pages/          ← latest release tag (or HEAD with --root-from-head)
#   site_pages/edge/     ← current working tree (main tip)
#
# Usage:
#   bash scripts/build_docs_pages.sh
#   bash scripts/build_docs_pages.sh --root-from-head   # local smoke: root = HEAD
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SITE_URL_BASE="${TORCHFITS_DOCS_URL:-https://astroai.github.io/torchfits}"
OUT="${TORCHFITS_DOCS_OUT:-$ROOT/site_pages}"
ROOT_FROM_HEAD=0
for arg in "$@"; do
  case "$arg" in
    --root-from-head) ROOT_FROM_HEAD=1 ;;
    -h|--help)
      sed -n '1,12p' "$0"
      exit 0
      ;;
  esac
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: missing command: $1" >&2
    exit 1
  }
}
need_cmd git
need_cmd python

if command -v zensical >/dev/null 2>&1; then
  ZENSICAL=(zensical)
elif command -v pixi >/dev/null 2>&1; then
  ZENSICAL=(pixi run --manifest-path "$ROOT/pixi.toml" zensical)
else
  echo "error: need zensical on PATH or pixi in this repo" >&2
  exit 1
fi

write_config() {
  local src_cfg=$1
  local dest=$2
  local site_url=$3
  local site_dir=$4
  local site_name=$5
  local channel=$6
  python - "$src_cfg" "$dest" "$site_url" "$site_dir" "$site_name" "$channel" <<'PY'
from pathlib import Path
import sys

src, dst, site_url, site_dir, site_name, channel = sys.argv[1:7]
lines: list[str] = []
seen_url = seen_dir = seen_name = False
for line in Path(src).read_text(encoding="utf-8").splitlines():
    if line.startswith("site_url"):
        lines.append(f'site_url = "{site_url}"')
        seen_url = True
    elif line.startswith("site_dir"):
        lines.append(f'site_dir = "{site_dir}"')
        seen_dir = True
    elif line.startswith("site_name"):
        lines.append(f'site_name = "{site_name}"')
        seen_name = True
    elif line.strip().startswith("docs_channel"):
        continue
    else:
        lines.append(line)
insert_at = 1
if not seen_name:
    lines.insert(insert_at, f'site_name = "{site_name}"')
    insert_at += 1
if not seen_url:
    lines.insert(insert_at, f'site_url = "{site_url}"')
    insert_at += 1
if not seen_dir:
    lines.insert(insert_at, f'site_dir = "{site_dir}"')

extra_idx = None
for i, line in enumerate(lines):
    if line.strip() == "[project.extra]":
        extra_idx = i
        break
if extra_idx is None:
    lines.extend(["", "[project.extra]", f'docs_channel = "{channel}"'])
else:
    lines.insert(extra_idx + 1, f'docs_channel = "{channel}"')
Path(dst).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
PY
}

build_tree() {
  local workdir=$1
  local site_url=$2
  local site_dir=$3
  local site_name=$4
  local channel=$5
  local dest=$6
  local cfg="$workdir/zensical.pages.toml"
  local src_cfg="$workdir/zensical.toml"
  if [[ ! -f "$src_cfg" ]]; then
    src_cfg="$ROOT/zensical.toml"
  fi

  write_config "$src_cfg" "$cfg" "$site_url" "$site_dir" "$site_name" "$channel"
  (
    cd "$workdir"
    if [[ -f scripts/sync_docs_examples.sh ]]; then
      bash scripts/sync_docs_examples.sh
    fi
    # Edge banner / CSS from tip-of-main even when building an older tag.
    if [[ "$workdir" != "$ROOT" ]]; then
      mkdir -p overrides docs/stylesheets
      cp -f "$ROOT/overrides/main.html" overrides/main.html
      if [[ -f "$ROOT/docs/stylesheets/extra.css" ]]; then
        cp -f "$ROOT/docs/stylesheets/extra.css" docs/stylesheets/extra.css
      fi
    fi
    rm -rf "$site_dir"
    "${ZENSICAL[@]}" build --clean -f zensical.pages.toml
  )
  mkdir -p "$dest"
  find "$dest" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  cp -a "$workdir/$site_dir"/. "$dest/"
  rm -f "$cfg"
  rm -rf "$workdir/$site_dir"
}

rm -rf "$OUT"
mkdir -p "$OUT"

LATEST_TAG="$(git describe --tags --abbrev=0 --match 'v*' 2>/dev/null || true)"
USE_HEAD_FOR_ROOT=$ROOT_FROM_HEAD
if [[ -z "$LATEST_TAG" ]]; then
  USE_HEAD_FOR_ROOT=1
fi

# Stable first so edge can overwrite /edge/ afterward.
if [[ "$USE_HEAD_FOR_ROOT" -eq 1 ]]; then
  echo "==> building stable docs from HEAD → $OUT/"
  build_tree "$ROOT" "${SITE_URL_BASE}/" "site_stable" "torchfits" "stable" "$OUT"
else
  echo "==> building stable docs from $LATEST_TAG → $OUT/"
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/torchfits-docs-stable.XXXXXX")"
  git worktree add --detach "$WORK" "$LATEST_TAG"
  # Use tip-of-main build tooling (zensical) against the tagged sources.
  build_tree "$WORK" "${SITE_URL_BASE}/" "site_stable" "torchfits" "stable" "$OUT"
  git worktree remove --force "$WORK"
fi

echo "==> building edge docs from tip of main → $OUT/edge/"
EDGE_SRC="$ROOT"
EDGE_WORK=""
# Prefer origin/main for /edge/ so a detached tag checkout cannot publish the
# tagged tree as "edge" (stable already comes from LATEST_TAG).
if git rev-parse --verify origin/main >/dev/null 2>&1; then
  if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
    EDGE_WORK="$(mktemp -d "${TMPDIR:-/tmp}/torchfits-docs-edge.XXXXXX")"
    git worktree add --detach "$EDGE_WORK" origin/main
    EDGE_SRC="$EDGE_WORK"
    echo "    (using origin/main at $(git -C "$EDGE_WORK" rev-parse --short HEAD))"
  fi
fi
build_tree "$EDGE_SRC" "${SITE_URL_BASE}/edge/" "site_edge" "torchfits (edge)" "edge" "$OUT/edge"
EDGE_SHA="$(git -C "$EDGE_SRC" rev-parse --short HEAD)"
if [[ -n "$EDGE_WORK" ]]; then
  git worktree remove --force "$EDGE_WORK"
fi

python - <<PY
from pathlib import Path
import datetime

out = Path("$OUT")
sha = "$EDGE_SHA"
tag = "$LATEST_TAG" or "none"
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
(out / "edge" / "EDGE_BUILD.txt").write_text(
    f"channel=edge\ncommit={sha}\nbuilt={now}\n", encoding="utf-8"
)
(out / "STABLE_BUILD.txt").write_text(
    f"channel=stable\ntag={tag}\nbuilt={now}\n", encoding="utf-8"
)
print(f"pages tree ready: {out}")
print(f"  stable tag: {tag}")
print(f"  edge commit: {sha}")
print(f"  edge URL:   ${SITE_URL_BASE}/edge/")
PY
