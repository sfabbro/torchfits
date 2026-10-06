"""`.cursor/hooks/harness-stop.sh` — the reflect-cadence bookkeeping.

The hook reads two JSON files: `harness/config.json` for the cadence thresholds
and `harness/state/stop-hook.json` for what it already did. Only the second was
guarded.

`config.json` was read straight into a Python one-liner. When that read dies,
`read -r ... < <(...)` gets no input, returns non-zero, and `set -e` aborts the
hook — *above* the state write. So a config.json with a trailing comma (or one
that is not JSON at all) left `lastProcessedGenerationId` frozen, which means the
`gen_id == last_gen` short-circuit never fires and, worse, `turnsSinceLastRun`
never accumulates: harness-reflect can never become due again until somebody
hand-repairs the file. Measured before the fix — both malformed variants exited
1 and left the state untouched, while a *missing* config was fine.

The hook's own comment already documents this exact failure for a different
input, one stanza further down, and explains that unfixed input made the state
write "never happen again". This is the same bug, still reachable.

These tests run the real hook in a temporary tree. `bash` is required, which is
also what the client executes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".cursor" / "hooks" / "harness-stop.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or not HOOK.is_file(),
    reason="the stop hook needs bash and the canonical .cursor tree",
)

PRIOR_STATE = {
    "lastRunAtMs": 1000,
    "turnsSinceLastRun": 3,
    "lastProcessedGenerationId": "gen-previous",
}
PAYLOAD = json.dumps(
    {
        "status": "completed",
        "loop_count": 0,
        "conversation_id": "conv-abcdef123456",
        "generation_id": "gen-new",
    }
)


@pytest.fixture()
def harness(tmp_path: Path) -> Path:
    """A throwaway ``.cursor/harness`` tree, and the hook's cwd."""
    (tmp_path / ".cursor" / "harness" / "state").mkdir(parents=True)
    (tmp_path / ".cursor" / "harness" / "trajectories").mkdir(parents=True)
    return tmp_path


def _state(root: Path) -> dict:
    return json.loads(
        (root / ".cursor" / "harness" / "state" / "stop-hook.json").read_text(
            encoding="utf-8"
        )
    )


def _seed(root: Path, config: str | None = None) -> None:
    (root / ".cursor" / "harness" / "state" / "stop-hook.json").write_text(
        json.dumps(PRIOR_STATE), encoding="utf-8"
    )
    if config is not None:
        (root / ".cursor" / "harness" / "config.json").write_text(
            config, encoding="utf-8"
        )


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(HOOK)], input=PAYLOAD, cwd=root, capture_output=True, text=True
    )


def _advanced(root: Path) -> bool:
    return _state(root).get("lastProcessedGenerationId") == "gen-new"


# --- the control ------------------------------------------------------------


def test_the_hook_advances_its_state_without_a_config(harness: Path) -> None:
    """Without this, every test below could pass by the hook simply never
    writing state at all."""
    _seed(harness)
    proc = _run(harness)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert _advanced(harness), _state(harness)


def test_a_valid_config_is_honoured(harness: Path) -> None:
    """A well-formed config must still reach the thresholds it names, so the
    guard added below is not a blanket 'ignore the file'."""
    _seed(harness, json.dumps({"reflect_min_turns": 99, "reflect_min_minutes": 99}))
    proc = _run(harness)
    assert proc.returncode == 0, proc.stderr[-800:]
    # turnsSinceLastRun is 4 after this call, and 4 < 99, so no reflect: the
    # thresholds really were read rather than defaulted to 5/45.
    state = _state(harness)
    assert state["turnsSinceLastRun"] == 4, state
    assert state["lastProcessedGenerationId"] == "gen-new", state
    assert "followup_message" not in proc.stdout, proc.stdout


# --- the finding ------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            '{"reflect_min_turns": 5, "reflect_min_minutes": 45,}', id="trailing-comma"
        ),
        pytest.param("reflect_min_turns = 5\n", id="not-json"),
        pytest.param("", id="empty"),
        pytest.param("[1, 2, 3]", id="wrong-type"),
    ],
)
def test_a_malformed_config_cannot_stop_the_state_write(
    harness: Path, config: str
) -> None:
    """R2-066. Each of these exited 1 and left lastProcessedGenerationId frozen,
    which stops both the duplicate-generation short-circuit and the reflect
    cadence from ever advancing again."""
    _seed(harness, config)
    proc = _run(harness)
    assert proc.returncode == 0, (
        f"a malformed config took the hook down; stderr:\n{proc.stderr[-800:]}"
    )
    assert _advanced(harness), (
        f"the hook exited cleanly but did not advance the state: {_state(harness)}"
    )


def test_a_malformed_config_falls_back_to_the_defaults(harness: Path) -> None:
    """Not 'ignore the file' -- fall back to 5 turns / 45 minutes, so a broken
    config still yields a working cadence rather than an inert hook."""
    _seed(harness, "{ not json")
    proc = _run(harness)
    assert proc.returncode == 0, proc.stderr[-800:]
    state = _state(harness)
    assert state["turnsSinceLastRun"] == 4, state


def test_the_hook_still_reflects_once_the_thresholds_are_met(harness: Path) -> None:
    """End-to-end proof that the cadence is live and not merely non-crashing:
    with turns at 4 and the default threshold of 5, the next completed
    conversation must trigger the reflect follow-up."""
    _seed(harness)
    _run(harness)  # turns 3 -> 4
    assert _state(harness)["turnsSinceLastRun"] == 4

    payload = json.loads(PAYLOAD)
    payload["generation_id"] = "gen-second"
    proc = subprocess.run(
        ["bash", str(HOOK)],
        input=json.dumps(payload),
        cwd=harness,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "followup_message" in proc.stdout, proc.stdout
    assert "harness-reflect" in proc.stdout, proc.stdout


def test_the_hook_stays_quiet_when_the_thresholds_are_not_met(harness: Path) -> None:
    _seed(harness, json.dumps({"reflect_min_turns": 99, "reflect_min_minutes": 99}))
    proc = _run(harness)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert proc.stdout.strip() == "{}", proc.stdout
