# Harness Playbook

## Bullets

- id: verify-tiers
  desc: verify_fast during edits; verify (ci-local) before push; verify_full only when human opts in — never torchregress-harness in the agent loop.

- id: ci-parity
  desc: GitHub CI lint+test matrix; local ci-local = preflight-push + pixi test.

- id: file-memory
  desc: Put durable notes in .cursor/harness/ — not long chat scrollback.

- id: minimal-diff
  desc: Smallest correct change; match existing repo style and tools.

- id: docs-api-sync
  desc: Root table helpers return dict[str,Tensor]; Arrow path is torchfits.table.*; never document env vars absent from src; integrity tests guard Core I/O signature fences + api.md env table.

- id: package-tree
  desc: Keep the git tree package-facing — Round-N scorecard CSVs under docs/assets/bench only when published; freeze audits go to .cursor/reviews/ (gitignored); no archive/ of agent dumps.

- id: pixi-test-env-rebuild
  desc: After C++/native edits, rebuild every pixi env you will run — `pixi run -e test -- pip install -e . --no-build-isolation` before pytest, and `pixi run -- pip install -e . --no-build-isolation` (or `pixi run dev`) before `pixi run python` example smoke; default and test envs do not share the extension.

- id: scorecard-deficit-significance
  desc: `compute_deficits` always emits lag rows; floors only set `significance` to noise|significant — unit tests must not expect `[]` for under-floor lags.

- id: int16-robust-quantize
  desc: Skewed float→int16 loss is write/quantize (BSCALE/BZERO or TSCAL/TZERO); use write(..., quantize="robust") / table.write quantize= — never default global min→max (poloka).

- id: cli-j-vs-J
  desc: CLI -j/--jobs = torch.set_num_threads; -J/--file-jobs = ThreadPool across files (each worker caps ATen to 1) — never fan out files with ATen alone.

- id: setkey-no-rewrite
  desc: setkey delete/rename must use CFITSIO fits_delete_key (+ binary copy for --out), never HDUList.write rewrite — rewrite decompresses CompImage and leaves stale Z* cards.

- id: cfitsio-http-ssrf
  desc: SSRF-check http/https/ftp via guard_fits_path/guard_cfitsio_remote_path before Core I/O; public URLs still open through CFITSIO network drivers (and Dataset/Range keep http_util). Case-fold sh:// in security.h.

- id: header-delitem-history
  desc: Header.__delitem__ routes through remove(..., remove_all=True) so HISTORY/COMMENT cards do not orphan after del.
