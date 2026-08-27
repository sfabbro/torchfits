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
  desc: guard_fits_path on I/O façades + torchfits.cpp path APIs before CFITSIO; put guards outside generator bodies and before optional imports (e.g. polars). Public http(s)/ftp still CFITSIO; private blocked. CLI is_remote_path includes ftp. CLI copy must guard then http_open (redirect-safe), never bare urlretrieve. Case-fold sh:// in security.h.

- id: unified-cache-stubs
  desc: C++ configure/clear/invalidate_cached are no-ops after Option A; SharedReadMeta is the live shared cache. Do not revive get_or_open_cached.

- id: header-delitem-history
  desc: Header.__delitem__ routes through remove(..., remove_all=True) so HISTORY/COMMENT cards do not orphan after del.

- id: where-prefer-mask
  desc: read_torch/table where= prefers project+torch-mask over read_fits_table_filtered gather; filtered is fallback only (dense predicates).

- id: bench-table-from-csv
  desc: Published bench summary tables must be recomputed from the cited results CSV (check units); never paste μs/ms from unrelated suites as MB/s.

- id: copy-is-binary
  desc: torchfits copy is shutil.copy2 (local) or http_open/urlretrieve (remote), not HDUList.write; CompImage stays compressed; refuse same-path; guard_fits_path before any fetch.

- id: tnull-read-torch
  desc: table.read_torch(where=) applies TNULL mask and _torch_cmp_mask (range-safe) before compare — same row set as Arrow table.read. Quantized TNULL is NaN in tensors; Arrow conversion with apply_fits_nulls maps those NaNs to null so IS NULL matches. Raw integer TNULL without TSCAL stays a sentinel.

- id: blank-nulval
  desc: BLANK present promotes the image to the scaled-float path (even identity BSCALE/BZERO) so nulval=NaN applies on full read and read_subset. HTTP Range treats BLANK as scaled (fallback to CFITSIO). Do not rewrite BSCALE=1 — that distorts real scale=1 packs. Native IEEE float/double must pass nullptr to fits_read_img (CFITSIO fnan treats Inf and signed zero as undefined when nulval is set); nulval=NaN is only for compressed tiles or integer storage read as float. Table TNULL→NaN only when TSCAL/TZERO cards exist (quantize); raw integer TNULL stays a sentinel. Unsigned/signed-byte conventions keep integer dtypes even with BLANK. Writing an already-decoded float with a copied integer header must drop BSCALE/BZERO/BLANK (CFITSIO would re-apply them).

- id: cpp-seal-all
  desc: torchfits._cpp.__getattr__ raises for names not in __all__; __dir__ is __all__ only. New _C symbols stay private until listed.

- id: kmp-duplicate-lib-ok
  desc: os.environ.setdefault KMP_DUPLICATE_LIB_OK=TRUE in __init__.py and pixi [activation.env]; required on macOS when torch and the extension both link libomp.
