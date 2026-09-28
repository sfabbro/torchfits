#!/usr/bin/env bash
# Stop hook: log trajectory stub + optionally trigger harness-reflect (ACE curator).
set -euo pipefail

input=$(cat)
HARNESS=".cursor/harness"
STATE="$HARNESS/state/stop-hook.json"
DEFAULT_MIN_TURNS=5
DEFAULT_MIN_MINUTES=45

mkdir -p "$HARNESS/trajectories" "$HARNESS/state"

# --- parse hook input ---
status=$(echo "$input" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status',''))")
loop_count=$(echo "$input" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('loop_count',0))")
conv_id=$(echo "$input" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('conversation_id','unknown'))")
gen_id=$(echo "$input" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('generation_id') or '')")

# --- load state ---
last_run=0
turns=0
last_gen=""
min_turns=$DEFAULT_MIN_TURNS
min_minutes=$DEFAULT_MIN_MINUTES

if [[ -f "$HARNESS/config.json" ]]; then
  read -r min_turns min_minutes < <(python3 -c "
import json
with open('$HARNESS/config.json') as f:
    c=json.load(f)
print(c.get('reflect_min_turns',$DEFAULT_MIN_TURNS), c.get('reflect_min_minutes',$DEFAULT_MIN_MINUTES))
")
fi

if [[ -f "$STATE" ]]; then
  read -r last_run turns last_gen < <(python3 -c "
import json
try:
  with open('$STATE') as f: s=json.load(f)
  print(s.get('lastRunAtMs',0), s.get('turnsSinceLastRun',0), s.get('lastProcessedGenerationId',''))
except Exception:
  print(0, 0, '')
")
fi

if [[ -n "$gen_id" && "$gen_id" == "$last_gen" ]]; then
  echo '{}'
  exit 0
fi

counted=0
if [[ "$status" == "completed" && "$loop_count" == "0" ]]; then
  counted=1
fi

now_ms=$(python3 -c "import time; print(int(time.time()*1000))")
turns=$((turns + counted))
minutes_since=999999
if [[ "$last_run" -gt 0 ]]; then
  minutes_since=$(( (now_ms - last_run) / 60000 ))
fi

# --- append lightweight trajectory marker ---
if [[ "$counted" -eq 1 && -d "$HARNESS" ]]; then
  day=$(date +%Y-%m-%d)
  stub="$HARNESS/trajectories/${day}-${conv_id:0:8}.md"
  if [[ ! -f "$stub" ]]; then
    cat >"$stub" <<EOF
# trajectory ${day}
conversation: ${conv_id}
status: ${status}
note: auto-stub from harness stop hook — agent may enrich after verify
EOF
  fi
fi

should_reflect=0
if [[ "$counted" -eq 1 && "$turns" -ge "$min_turns" && "$minutes_since" -ge "$min_minutes" ]]; then
  should_reflect=1
fi

# Every value crosses into Python through the environment, and the heredoc is
# quoted, so nothing from the hook's JSON input is ever spliced into source
# text. generation_id came from stdin as free-form JSON: a value containing a
# double quote produced a SyntaxError here, and with `set -e` the state file
# below was then never written, so the reflect cadence lost its bookkeeping
# permanently. harness-session-start.sh already passes its one value this way
# (HARNESS_PLAYBOOK); this keeps the two hooks consistent.
HARNESS_GEN_ID="$gen_id" HARNESS_NOW_MS="$now_ms" HARNESS_LAST_RUN="$last_run" \
  HARNESS_TURNS="$turns" HARNESS_REFLECT="$should_reflect" HARNESS_STATE="$STATE" \
  python3 - <<'PY'
import json, os

now_ms = int(os.environ["HARNESS_NOW_MS"])
last_run = int(os.environ["HARNESS_LAST_RUN"])
turns = int(os.environ["HARNESS_TURNS"])
should_reflect = os.environ["HARNESS_REFLECT"] == "1"
state = {
  "lastRunAtMs": now_ms if should_reflect else last_run,
  "turnsSinceLastRun": 0 if should_reflect else turns,
  "lastProcessedGenerationId": os.environ.get("HARNESS_GEN_ID", ""),
}
with open(os.environ["HARNESS_STATE"], "w") as f:
  json.dump(state, f, indent=2)
PY

if [[ "$should_reflect" -eq 1 ]]; then
  python3 -c 'import json; print(json.dumps({"followup_message": "Run the harness-reflect skill now on .cursor/harness/. If failures/ has 3+ recent pattern files, run harness-improve next. If no playbook updates, respond exactly: No playbook updates."}))'
else
  echo '{}'
fi
