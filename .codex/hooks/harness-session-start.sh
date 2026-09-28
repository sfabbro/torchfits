#!/usr/bin/env bash
# Inject playbook context at session start when .cursor/harness exists.
set -euo pipefail

input=$(cat)
playbook=".cursor/harness/playbook.md"

if [[ ! -f "$playbook" ]]; then
  echo '{}'
  exit 0
fi

export HARNESS_PLAYBOOK="$playbook"
python3 <<'PY'
import json, os, re

path = os.environ.get("HARNESS_PLAYBOOK", ".cursor/harness/playbook.md")
try:
    text = open(path).read()
except OSError:
    print("{}")
    raise SystemExit(0)

# ponytail: cap at 12 bullets to limit context injection
ids = re.findall(r"^- id:\s*(\S+)\s*\n\s*desc:\s*(.+)$", text, re.M)
if not ids:
    print("{}")
    raise SystemExit(0)

lines = [f"- id: {i}\n  desc: {d.strip()}" for i, d in ids[:12]]
msg = (
    "Harness playbook (full file: .cursor/harness/playbook.md). "
    "Follow harness-coding: plan → execute → verify → file memory.\n\n"
    + "\n\n".join(lines)
)
print(json.dumps({"additional_context": msg}))
PY
