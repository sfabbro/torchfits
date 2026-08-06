#!/usr/bin/env bash
# Copy runnable examples into docs/ so the published site can link them.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/docs/published-examples"
rm -rf "${DEST}"
mkdir -p "${DEST}/cli"
shopt -s nullglob
for f in "${ROOT}/examples"/*.py; do
  cp "$f" "${DEST}/"
done
for f in "${ROOT}/examples/cli"/*; do
  [[ -f "$f" ]] || continue
  cp "$f" "${DEST}/cli/"
done
# Marker so empty trees are obvious in CI logs; the generated README is not a
# source file (docs/published-examples/ is uncommitted), so hide the Edit /
# View source buttons that would 404 on raw/main.
printf -- '---\nhide:\n  - edit\n  - view\n---\n\n# Published examples\n\nCopied from `examples/` at docs-build time. Do not edit.\n' \
  > "${DEST}/README.md"
