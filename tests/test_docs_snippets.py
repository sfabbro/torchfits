"""The multi-worker `DataLoader` snippet in `docs/quickstart.md` must run.

`docs/quickstart.md` is the first page a reader opens, and section 8 builds a
`FitsImageDataset` + `make_loader(..., num_workers=4)` pipeline and iterates it.
Run verbatim as a script on macOS that snippet dies:

    RuntimeError: DataLoader worker (pid 73282) exited unexpectedly with exit
    code 1. Details are lost due to multiprocessing.

The cause is the standard `spawn` hazard, and it is silent: macOS has used the
`spawn` start method by default since Python 3.8, each worker re-imports the
`__main__` module, the unguarded top-level loop builds a *second* loader in
every worker, and the recursive spawn takes the process down with an error that
names nothing about the cause. Measured on the snippet exactly as the page
wrote it (rc=1); adding the `if __name__ == "__main__":` guard the page now
carries gives rc=0 and prints the batch.

Nothing in `tests/test_docs_integrity.py` executes a docs Python fence -- it
parses CLI commands, checks API member names and parameter-table completeness,
and compares benchmark claims to their runs. A snippet that cannot run is
invisible to all of that, which is why this page shipped the unguarded form for
as long as it did.

Both tests here are worth having separately:

* the first **runs** the page's own text, so removing the guard fails it;
* the second is a structural guard over every docs page, so a *new*
  multi-worker snippet cannot be added without the guard.

The repository's own examples are the house style this restores: all eight
`examples/` scripts that build a loader carry the guard.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# ```python ... ``` fences, indented or not (admonition/tab bodies indent them).
PY_FENCE = re.compile(
    r"^[ \t]*```(?:python|py)[ \t]*\n(.*?)^[ \t]*```[ \t]*$", re.S | re.M
)
MULTI_WORKER = re.compile(r"num_workers\s*=\s*[1-9]")
ITERATES_LOADER = re.compile(r"for\s+[\w,\s]*?\bin\s+\w*loader\b")
# Quote-agnostic: a reformat to single quotes must not read as "no guard".
MAIN_GUARD = re.compile(r"""__name__\s*==\s*["']__main__["']""")


def _fences(page: str) -> list[str]:
    return PY_FENCE.findall((DOCS / page).read_text(encoding="utf-8"))


def _fences_that_iterate_a_multi_worker_loader() -> list[tuple[str, int, str]]:
    hits = []
    for page in sorted(DOCS.glob("*.md")):
        for i, block in enumerate(_fences(page.name), 1):
            if MULTI_WORKER.search(block) and ITERATES_LOADER.search(block):
                hits.append((page.name, i, textwrap.dedent(block)))
    return hits


@pytest.mark.skipif(
    sys.platform != "darwin", reason="the spawn hazard is macOS's default"
)
def test_quickstart_multi_worker_snippet_runs_verbatim(tmp_path: Path) -> None:
    """The page's own text, run as a script, must exit 0.

    Extracting the fence rather than restating it is the point: a fix applied to
    this file rather than to `docs/quickstart.md` passes while the reader still
    sees the broken version.
    """
    import numpy as np

    import torchfits

    blocks = [
        block
        for block in _fences("quickstart.md")
        if MULTI_WORKER.search(block) and ITERATES_LOADER.search(block)
    ]
    assert len(blocks) == 1, (
        f"expected exactly one iterating multi-worker snippet in quickstart.md, "
        f"found {len(blocks)}; this test needs its target named explicitly"
    )

    survey = tmp_path / "data" / "survey"
    survey.mkdir(parents=True)
    rng = np.random.default_rng(0)
    image = rng.random((64, 64), dtype=np.float32)
    for i in range(6):
        torchfits.write(
            str(survey / f"part_{i}.fits"),
            image,
            overwrite=True,
            header={"CLASS_ID": i},
        )

    script = tmp_path / "snippet.py"
    script.write_text(blocks[0], encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, (
        "docs/quickstart.md's multi-worker DataLoader snippet does not run as "
        "written. The workers are started with `spawn`, which re-imports the "
        f'module, so the snippet needs `if __name__ == "__main__":` around the\n'
        f"training loop.\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def test_every_docs_multi_worker_snippet_carries_the_main_guard() -> None:
    """No page may iterate a `num_workers>0` loader outside `__main__`.

    Constructing a loader does not spawn anything -- `torch.utils.data.DataLoader`
    starts its workers on the first `__iter__` -- so only the iterating snippets
    are affected. That makes this guard cheap and exact: the four snippets that
    merely build a loader (`api-data.md` x2, `examples-ml.md` x2) were measured
    at rc=0 as written and are correctly left alone.
    """
    hits = _fences_that_iterate_a_multi_worker_loader()
    assert hits, "no docs snippet iterates a multi-worker loader; the scan is broken"

    offenders = [
        f"docs/{page} fence#{i}"
        for page, i, block in hits
        if not MAIN_GUARD.search(block)
    ]
    assert not offenders, (
        "these snippets iterate a DataLoader with num_workers > 0 at module "
        "top level, so every worker re-imports the module, builds a second "
        'loader and dies. Wrap the loop in `if __name__ == "__main__":`:\n'
        + "\n".join(offenders)
    )


def test_the_house_examples_that_build_a_loader_use_the_guard() -> None:
    """The docs fix matches what `examples/` already does.

    If a future change drops the guard from every example script, the docs are
    no longer the outlier and this test should fail rather than let the two
    drift apart quietly.
    """
    guarded = unguarded = 0
    for path in sorted((ROOT / "examples").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "make_loader(" not in text and "DataLoader(" not in text:
            continue
        if MAIN_GUARD.search(text):
            guarded += 1
        else:
            unguarded += 1
            pytest.fail(f"{path.name} builds a loader without a __main__ guard")
    assert guarded >= 8, f"only {guarded} loader examples carry the guard"
