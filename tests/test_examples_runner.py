"""Pytest wrapper for runnable example scripts."""

from __future__ import annotations

import ast
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples" / "test_examples.py"
SAMPLE_DATA = ROOT / "examples" / "_sample_data.py"


def _load_module(path: Path, name: str):
    """Import ``path`` under ``name``, bypassing the package machinery.

    ``examples/`` is not an installed package, and _sample_data.py binds
    CACHE_DIR at import time, so every test needs its own module object with
    its own cache root.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sample_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """examples/_sample_data with TORCHFITS_SAMPLE_CACHE pointed at tmp_path."""
    monkeypatch.setenv("TORCHFITS_SAMPLE_CACHE", str(tmp_path / "samples"))
    return _load_module(SAMPLE_DATA, f"_torchfits_sample_data_{tmp_path.name}")


def _parsed_calls(source: str) -> dict[str, ast.Call]:
    """Map callee name -> call node for every call in ``source``.

    Inspecting parsed call sites rather than grepping the text matters here:
    a substring check for "urlretrieve" also matches the comment explaining
    why urlretrieve is not used, and would pass vacuously.
    """
    calls: dict[str, ast.Call] = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                calls[func.attr] = node
            elif isinstance(func, ast.Name):
                calls[func.id] = node
    return calls


def test_example_scripts_exit_zero() -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"examples/test_examples.py failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_network_examples_bound_their_downloads() -> None:
    """Network fetches in required examples must be time-bounded.

    Regression: the Galaxy Zoo example is required (it is not in the runner's
    OPTIONAL set) and pulls one FITS cutout per row from a third-party HTTP
    service, but used ``urllib.request.urlretrieve``, which takes no timeout and
    blocks until the OS gives up. Measured against an unroutable address, a
    single cutout took 75.0s that way, so a cold cache with the runner's GZ_N=8
    could spend 600s inside an example whose runner budget is 300s -- and the
    runner reports a timeout as a failure. With an explicit per-cutout timeout a
    slow cutout is skipped by the handler that was already there, and the
    remaining budget covers the ones that did arrive.
    """
    source = (ROOT / "examples" / "example_ml_galaxyzoo_legacy.py").read_text()
    calls = _parsed_calls(source)

    assert "urlretrieve" not in calls, (
        "urlretrieve() cannot time out; use urlopen() with an explicit timeout"
    )
    assert "urlopen" in calls, "the cutout fetch should go through urlopen()"
    timeout_kw = calls["urlopen"].keywords
    assert any(kw.arg == "timeout" for kw in timeout_kw), (
        "urlopen() must be given a timeout, or a stalled cutout hangs the example"
    )
    # The handler must already tolerate a timeout, or adding one turns a slow
    # cutout into a hard failure instead of a skip.
    assert issubclass(TimeoutError, OSError), "expected platform invariant"
    assert "except (urllib.error.URLError, OSError):" in source, (
        "a timed-out cutout must fall into the existing skip handler"
    )


def _load_runner():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_torchfits_examples_runner", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("script_body", "name", "is_optional", "expected_ok"),
    [
        # An optional example that declines to run passes rather than failing.
        (
            'print("SKIP: no network")\nraise SystemExit(1)\n',
            "example_polars.py",
            True,
            True,
        ),
        # ... and so does one that is optional purely because it needs the
        # network, which is the Galaxy Zoo case.
        (
            'print("SKIP: galaxy_zoo1_table2 sample unavailable (no network)")\n'
            "raise SystemExit(1)\n",
            "example_ml_galaxyzoo_legacy.py",
            True,
            True,
        ),
        # A required example in the same situation still fails: a repo-side
        # problem must not be excused by the skip convention.
        (
            'print("SKIP: something")\nraise SystemExit(1)\n',
            "example_required.py",
            False,
            False,
        ),
        # A required example that genuinely fails, with no skip message.
        ('print("boom")\nraise SystemExit(3)\n', "example_required.py", False, False),
        # Optional and happy still counts as a pass.
        ('print("fine")\n', "example_polars.py", True, True),
        # An indented SKIP: the examples print these from inside `if` blocks.
        (
            'if True:\n    print("SKIP: sample not cached")\nraise SystemExit(1)\n',
            "example_ml_galaxyzoo_legacy.py",
            True,
            True,
        ),
    ],
)
def test_optional_examples_may_decline_without_failing_the_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script_body: str,
    name: str,
    is_optional: bool,
    expected_ok: bool,
) -> None:
    """OPTIONAL must mean "cannot red the gate", for every decline reason.

    Regression: the runner only honoured a two-entry marker list ("not
    installed", "skipping") that matched none of the seven examples printing the
    repo's actual "SKIP: ..." convention, so an optional example that declined to
    run was still reported as a failure.
    """
    runner = _load_runner()
    script = tmp_path / "fake_example.py"
    script.write_text(script_body)

    optional = set(runner.OPTIONAL)
    if is_optional:
        optional.add(name)
    else:
        optional.discard(name)
    monkeypatch.setattr(runner, "OPTIONAL", optional)
    monkeypatch.setattr(runner, "_example_path", lambda _n: str(script))
    ok, _detail = runner._run_example(name)
    assert ok is expected_ok


def test_optional_example_timeout_is_a_skip_but_a_required_one_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timeout must not red the gate for an optional example.

    This is the case CP-012 left open. Bounding each cutout download removed the
    indefinite hang, but the runner still had a wall-clock budget, and a timeout
    was reported as a failure for optional and required examples alike -- so a
    slow network could still fail CI for an example whose entire purpose is to
    demonstrate the network path.
    """
    runner = _load_runner()
    script = tmp_path / "slow_example.py"
    script.write_text("import time\ntime.sleep(30)\n")

    monkeypatch.setattr(runner, "_example_path", lambda _n: str(script))
    monkeypatch.setattr(runner, "OPTIONAL", {"example_slow_optional.py"})
    # Both names need the short budget, or the required case silently runs to
    # completion on the 180s default and passes for the wrong reason.
    monkeypatch.setitem(runner.TIMEOUTS, "example_slow_optional.py", 1)
    monkeypatch.setitem(runner.TIMEOUTS, "example_slow_required.py", 1)

    ok, detail = runner._run_example("example_slow_optional.py")
    assert ok is True, f"optional example timed out but should skip: {detail}"
    assert "timed out" in detail

    # The same timeout on a required example is still a failure.
    ok, detail = runner._run_example("example_slow_required.py")
    assert ok is False
    assert "timeout after 1s" in detail


# ---------------------------------------------------------------------------
# EX-001 / EX-002 -- examples/_sample_data.py, the helper every sample-backed
# example routes through. The urlretrieve timeout defect was already fixed (and
# guarded) for the one example that fetches on its own; the shared helper kept
# it, so a slow network could still blow a required example's runner budget.
# ---------------------------------------------------------------------------


def test_shared_sample_helper_fetches_are_time_bounded() -> None:
    """Every download in the shared sample helper must be able to time out.

    Regression: ``ensure_sample`` used ``urllib.request.urlretrieve``, which
    takes no timeout and blocks until the OS gives up. Measured against an
    unroutable address, one sample took 75.0s that way. ``example_m13_stack``
    needs five of them in sequence (375s) against a 180s per-example budget and
    is *required*, so the runner reports a timeout as FAIL rather than a skip;
    ``example_lupton_rgb_sdss`` needs three (225s). With a bounded timeout each
    failure raises ``SampleUnavailable``, ``try_ensure_sample`` returns None and
    the example prints SKIP and exits 0.
    """
    calls = _parsed_calls(SAMPLE_DATA.read_text(encoding="utf-8"))

    assert "urlretrieve" not in calls, (
        "urlretrieve() cannot time out; use urlopen() with an explicit timeout"
    )
    assert "urlopen" in calls, "the sample fetch should go through urlopen()"
    assert any(kw.arg == "timeout" for kw in calls["urlopen"].keywords), (
        "urlopen() must be given a timeout, or a stalled sample hangs the example"
    )
    assert issubclass(TimeoutError, OSError), "expected platform invariant"
    # The handler must already tolerate a timeout, or adding one turns a slow
    # sample into a hard failure instead of a skip.
    assert "except (urllib.error.URLError, OSError) as exc:" in SAMPLE_DATA.read_text(
        encoding="utf-8"
    ), "a timed-out sample must fall into the existing SampleUnavailable handler"


def _fake_fits(nbytes: int = 20_000) -> bytes:
    """A payload with a FITS magic and at least one 2880-byte block."""
    return b"SIMPLE  =".ljust(nbytes, b"\0")


@pytest.mark.parametrize(
    ("name", "magic"),
    [
        ("horsehead", b"SIMPLE  ="),
        ("sdss_lupton_g", b"BZh"),
        ("manga_logcube", b"\x1f\x8b"),
    ],
)
def test_each_sample_container_is_validated_by_its_own_magic(
    sample_data, name: str, magic: bytes
) -> None:
    """A bzip2/gzip sample must not be judged by the plain-FITS rule.

    ``_dest_path`` keeps the URL's compound suffix, so three of the sixteen
    samples are compressed. A guard that only knows ``SIMPLE  =`` would reject
    every real bzip2 and gzip sample in the table and each of the examples
    that uses it would skip forever.
    """
    dest = sample_data._dest_path(name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(magic.ljust(20_000, b"\0"))

    assert sample_data._is_sample_file(name, dest) is True
    assert sample_data.ensure_sample(name, allow_download=False) == dest


def test_an_unusable_cached_sample_is_evicted_not_served(
    sample_data, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated cache entry must not be handed to the examples forever.

    Regression: the cache-hit test was ``st_size > 0``, so a 70-byte garbage
    file was returned as a valid sample. Every example that opened it then died
    with "Could not open FITS file" -- and because the file was in the cache,
    nothing ever re-fetched it, so the failure was permanent and undiagnosed.

    FAST mode is pinned so the eviction path is exercised without reaching the
    network: the examples only ever see the post-eviction behaviour.
    """
    monkeypatch.setenv("TORCHFITS_EXAMPLE_FAST", "1")
    dest = sample_data._dest_path("horsehead")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"SIMPLE  =".ljust(70, b"\0"))

    with pytest.raises(sample_data.SampleUnavailable) as excinfo:
        sample_data.ensure_sample("horsehead", allow_download=False)
    assert "removed an unusable 70-byte cached copy" in str(excinfo.value)
    assert not dest.exists(), "the unusable entry must be evicted, not left in place"
    assert sample_data.try_ensure_sample("horsehead") is None


def test_an_unusable_cached_sample_is_replaced_by_a_fresh_fetch(
    sample_data, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The eviction must self-heal: the next call downloads a good copy.

    This is the path a user whose fetch was cut short actually lands on, and it
    is the recovery the old code never offered.
    """
    monkeypatch.delenv("TORCHFITS_EXAMPLE_FAST", raising=False)
    dest = sample_data._dest_path("horsehead")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"SIMPLE  =".ljust(70, b"\0"))
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_a, **_k: _FakeResponse(_fake_fits(50_000))
    )

    assert sample_data.ensure_sample("horsehead") == dest
    assert dest.stat().st_size == 50_000
    assert dest.read_bytes().startswith(b"SIMPLE  =")


def test_a_valid_cached_sample_is_still_served(sample_data) -> None:
    """Positive control: the guard must not reject a good cache entry."""
    dest = sample_data._dest_path("horsehead")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_fake_fits())

    assert sample_data.ensure_sample("horsehead", allow_download=False) == dest
    assert dest.is_file(), "a valid entry must not be evicted"


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._buf = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:  # shutil.copyfileobj contract
        return self._buf.read(size)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


@pytest.mark.parametrize(
    ("payload", "label"),
    [
        (b"SIMPLE  =".ljust(1_000, b"\0"), "truncated below one FITS block"),
        (
            b"<html><body>502 Bad Gateway</body></html>".ljust(20_000, b" "),
            "error page",
        ),
        (b"".ljust(50_000, b"\0"), "all zeros"),
    ],
    ids=["short", "error-page", "zeros"],
)
def test_a_short_or_wrong_transfer_is_never_committed_to_the_cache(
    sample_data, monkeypatch: pytest.MonkeyPatch, payload: bytes, label: str
) -> None:
    """The .partial must be validated before it replaces the cache entry.

    ``urlretrieve`` does not raise when a server closes a response early, so
    the old code could commit a truncated file -- which the old cache-hit test
    then accepted forever. The two defects compounded.
    """
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_a, **_k: _FakeResponse(payload)
    )

    with pytest.raises(sample_data.SampleUnavailable) as excinfo:
        sample_data.ensure_sample("horsehead")
    assert "horsehead" in str(excinfo.value), label
    assert not sample_data._dest_path("horsehead").exists(), label
    assert not list(sample_data.CACHE_DIR.glob("*.partial")), (
        f"the rejected {label} .partial must be cleaned up"
    )


def test_a_good_transfer_is_streamed_to_disk_and_committed(
    sample_data, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control for the streaming fetch path."""
    # CI sets this for the example runner. This test is the download path.
    monkeypatch.delenv("TORCHFITS_EXAMPLE_FAST", raising=False)
    seen: dict[str, object] = {}

    def fake_urlopen(url, timeout=None):  # noqa: ANN001, ANN202
        seen["url"] = url
        seen["timeout"] = timeout
        return _FakeResponse(_fake_fits(50_000))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    dest = sample_data.ensure_sample("horsehead")
    assert dest.is_file()
    assert dest.read_bytes() == _fake_fits(50_000)
    assert seen["url"] == sample_data.SAMPLES["horsehead"]
    assert seen["timeout"] == sample_data.SAMPLE_TIMEOUT_S
    assert sample_data.SAMPLE_TIMEOUT_S > 0, "a zero/absent timeout is not a bound"


# ---------------------------------------------------------------------------
# EX-003 / EX-005 -- examples/test_examples.py, the gate itself.
# ---------------------------------------------------------------------------


def test_discovery_survives_a_checkout_path_containing_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whether a hit is a cli/ script is decided by the glob, not the path text.

    Regression: the classifier was ``"cli" not in pattern`` applied to the
    *absolute* glob, so any checkout whose path contains "cli" (~/clinical/...,
    ~/client/..., a worktree named cli-work) reclassified every top-level
    example as ``cli/<name>.py``. ``_example_path`` then resolved to a
    non-existent file and all 32 required examples reported "file not found".
    """
    runner = _load_runner()
    for parent in ("examples", "clinical/examples", "client/examples", "cli/examples"):
        root = tmp_path / parent
        (root / "cli").mkdir(parents=True)
        (root / "example_image.py").write_text("", encoding="utf-8")
        (root / "_helper.py").write_text("", encoding="utf-8")
        (root / "test_examples.py").write_text("", encoding="utf-8")
        (root / "cli" / "make_rgb_demo.py").write_text("", encoding="utf-8")

        monkeypatch.setattr(runner, "SCRIPT_DIR", str(root))
        assert runner._discover_examples() == ["example_image.py"], parent
        # Every discovered name must resolve to a file that exists, or the
        # runner reports "file not found" for the whole gate.
        for name in runner._discover_examples():
            assert (root / name).is_file(), f"{parent}: {name}"


def test_the_runner_does_not_hand_pyythonoptimize_to_the_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PYTHONOPTIMIZE erases every assert in the self-checking examples.

    Regression: ``_run_example`` passed ``os.environ.copy()`` straight through,
    so a developer or CI image with PYTHONOPTIMIZE=1 ran
    example_identity_stress.py (20 asserts), example_cfitsio_cookbook.py (18)
    and example_ccfits_cookbook.py (26) with every assertion removed. All three
    still print "All ... checks passed" and exit 0, and the gate reports PASS.
    Verified: with read_tensor patched to corrupt values, the normal run raises
    AssertionError and the -O run prints "All identity checks passed".
    """
    runner = _load_runner()
    report = tmp_path / "report.txt"
    script = tmp_path / "child.py"
    script.write_text(
        "import os\n"
        f"open({str(report)!r}, 'w').write(\n"
        "    f\"{int(__debug__)}|{os.environ.get('PYTHONOPTIMIZE', '')!r}\")\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    monkeypatch.setattr(runner, "_example_path", lambda _n: str(script))
    monkeypatch.setitem(runner.TIMEOUTS, "example_child.py", 60)

    ok, detail = runner._run_example("example_child.py")
    assert ok is True, detail
    child_debug, child_env = report.read_text(encoding="utf-8").split("|")
    assert child_env == "''", f"PYTHONOPTIMIZE reached the example process: {child_env}"
    assert child_debug == "1", "the example process ran with assertions disabled"

    # And the guard that covers the runner's own interpreter: -O here cannot be
    # undone for the children, so the gate must refuse instead of reporting a
    # green run that verified nothing.
    assert runner._assertions_enabled() is True
    monkeypatch.setattr(runner, "_assertions_enabled", lambda: False)
    ok, detail = runner._run_example("example_child.py")
    assert ok is False
    assert "assertions disabled" in detail


# ---------------------------------------------------------------------------
# EX-004 -- the examples must fail when the claim they print is false.
# ---------------------------------------------------------------------------

_BREAK_DRIVER = """
import runpy, sys, torch, torchfits

target, mode, example = sys.argv[1], sys.argv[2], sys.argv[3]
_real = getattr(torchfits, target)
if mode == "corrupt_write":
    def patched(path, data, *a, **k):
        _real(path, data, *a, **k)
        _real(path, data * 1.5, *a, **k)
else:
    def patched(path, *a, **k):
        return _real(path, *a, **k) + 1000.0
setattr(torchfits, target, patched)
runpy.run_path(example, run_name="__main__")
"""


@pytest.mark.parametrize(
    ("example", "target", "mode", "success_line"),
    [
        (
            "example_image.py",
            "write_tensor",
            "corrupt_write",
            "write_tensor round-trip: True",
        ),
        (
            "example_image_cutouts.py",
            "read_subset",
            "corrupt",
            "tensor slice matches read_subset: True",
        ),
    ],
)
def test_an_example_that_prints_a_failed_check_must_not_exit_zero(
    tmp_path: Path, example: str, target: str, mode: str, success_line: str
) -> None:
    """Printing a correctness boolean is not a check.

    Regression: example_image.py printed ``write_tensor round-trip: False`` and
    example_image_cutouts.py printed two ``... matches ...: False`` lines on a
    broken read/write path, then returned 0 -- so the smoke gate reported PASS
    while the example had just demonstrated a data-loss bug. Both now raise.
    """
    driver = tmp_path / "driver.py"
    driver.write_text(_BREAK_DRIVER, encoding="utf-8")

    env = dict(os.environ)
    env.update(
        {
            "TORCHFITS_EXAMPLE_FAST": "1",
            "TORCHFITS_SAMPLE_CACHE": str(tmp_path / "empty_cache"),
            "KMP_DUPLICATE_LIB_OK": "TRUE",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            str(driver),
            target,
            mode,
            str(ROOT / "examples" / example),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, (
        f"{example} exited 0 with a broken {target} -- it printed the mismatch "
        f"instead of failing\nstdout:\n{result.stdout}"
    )
    assert success_line not in result.stdout, (
        f"{example} reached its success line with a broken {target}:\n{result.stdout}"
    )
    assert ": False" not in result.stdout, (
        f"{example} printed a failed check and still exited 0:\n{result.stdout}"
    )
    assert "RuntimeError" in result.stderr, (
        f"{example} should fail loudly on a broken {target}:\n{result.stderr}"
    )


def test_an_example_that_prints_a_failed_check_still_passes_when_it_is_honest(
    tmp_path: Path,
) -> None:
    """Positive control: the two examples must still exit 0 on an unpatched run.

    Without this, a test that always saw a non-zero exit would pass even if the
    examples were broken for an unrelated reason.
    """
    env = dict(os.environ)
    env.update(
        {
            "TORCHFITS_EXAMPLE_FAST": "1",
            "TORCHFITS_SAMPLE_CACHE": str(tmp_path / "empty_cache"),
            "KMP_DUPLICATE_LIB_OK": "TRUE",
        }
    )
    for example, success_line in (
        ("example_image.py", "write_tensor round-trip: True"),
        ("example_image_cutouts.py", "tensor slice matches read_subset: True"),
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "examples" / example)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{example}:\n{result.stderr}"
        assert success_line in result.stdout, f"{example}:\n{result.stdout}"


# ---------------------------------------------------------------------------
# EX-00x: a skip marker found anywhere in the output laundered a real crash
# into a PASS (round-2 unit 18).
#
# `_run_example` matched SKIP_MARKERS as bare substrings against the whole
# combined stdout+stderr. So an optional example that died hard -- printing a
# traceback, or an unrelated warning -- was reported PASS (optional) as long as
# its output contained the word "skipping" or "not installed" *anywhere*, and
# the diagnostic was thrown away.
#
# No example relies on the loose match: every genuine decline in the tree is
# "SKIP: ..." at the start of a line, and the two loose markers appear only
# mid-line on paths that print and then continue (exit 0, so they never reach
# this check). Requiring line-start keeps every real decline a skip.
# ---------------------------------------------------------------------------


def _run_optional(runner, tmp_path, monkeypatch, body: str) -> tuple[bool, str]:
    script = tmp_path / "fake_optional.py"
    script.write_text(body)
    monkeypatch.setattr(runner, "_example_path", lambda _n: str(script))
    monkeypatch.setattr(runner, "OPTIONAL", {"fake_optional.py"})
    return runner._run_example("fake_optional.py")


def test_a_hard_crash_naming_a_skip_word_is_not_a_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finding. The path in the diagnostic contains "skipping"."""
    runner = _load_runner()
    body = (
        "import sys, json\n"
        'sys.stderr.write(json.dumps({"file": "/tmp/skipping/data.fits"}) + "\\n")\n'
        'sys.stderr.write("ValueError: corrupt primary\\n")\n'
        "raise SystemExit(1)\n"
    )
    ok, detail = _run_optional(runner, tmp_path, monkeypatch, body)
    assert ok is False, f"a hard crash was laundered into a skip: {detail!r}"
    # The diagnostic must survive: the whole point of failing is that a human
    # can read why.
    assert "corrupt primary" in detail, detail


def test_an_unrelated_not_installed_warning_does_not_excuse_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warning about an unrelated missing package, then a hard failure."""
    runner = _load_runner()
    body = (
        "import sys, warnings\n"
        'warnings.warn("scipy not installed; falling back to a lossy path")\n'
        'sys.stderr.write("AssertionError: expected 3 rows, got 0\\n")\n'
        "raise SystemExit(1)\n"
    )
    ok, detail = _run_optional(runner, tmp_path, monkeypatch, body)
    assert ok is False, f"laundered: {detail!r}"
    assert "expected 3 rows" in detail, detail


def test_a_line_starting_marker_is_still_a_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control. Without it the two tests above could pass by rejecting
    every decline -- which would re-break the defect the markers were added
    for."""
    runner = _load_runner()
    for body in (
        'print("SKIP: polars is not installed")\nraise SystemExit(1)\n',
        'print("skipping: no network")\nraise SystemExit(1)\n',
        'print("not installed")\nraise SystemExit(1)\n',
        'if True:\n    print("  SKIP: sample not cached")\nraise SystemExit(1)\n',
    ):
        ok, detail = _run_optional(runner, tmp_path, monkeypatch, body)
        assert ok is True, (
            f"a genuine decline stopped being a skip: {body!r} -> {detail!r}"
        )


def test_declined_requires_a_line_to_begin_with_the_marker() -> None:
    """`_declined` directly, so the rule is pinned apart from the runner."""
    runner = _load_runner()
    assert runner._declined("SKIP: no sample")
    assert runner._declined("    SKIP: indented, and the examples indent")
    assert runner._declined("noise\nSKIP: on the second line")
    assert runner._declined("case-insensitive\nskip: lower")
    assert not runner._declined("Traceback (most recent call last):")
    assert not runner._declined("reading /tmp/skipping/data.fits")
    assert not runner._declined("scipy not installed; then it crashed")
    assert not runner._declined("")
