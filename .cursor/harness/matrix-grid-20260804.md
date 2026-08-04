# CANFAR matrix grid — 2026-08-04 (full run)

## Objective
Full-grid bench: Python 3.10–3.14 × torch lanes 2.10–2.13 × {CPU, CUDA}
on CANFAR staging via the pixi-first in-container flow. 41 legs; all
completed successfully; results fetched locally + VOS + ARC.

## Grid composition (41 legs, 151,691 result rows)
- CPU 20 legs × 3,057 rows (5 py × 4 lanes)
- CUDA 21 legs × 4,299–4,315 rows (5 py × 4 lanes + cu130 spot on py3.13/t2.13)

All bundles: `benchmarks_results/exhaustive_matrix_<tag>_<ts>/`
(results.csv, fitstable_results.csv, megacam_results.csv ×160,
torchfits_deficits.csv, environment.txt, summary.md, _raw/).
VOS: `vos:sfabbro/torchfits-gpu-bench/<run_id>`; ARC mirror under
`/arc/home/sfabbro/torchfits-gpu-bench/<run_id>`.
Aggregate: `benchmarks_results/matrix_grid_20260804_aggregate.csv`.

## Failures found & fixed during the run (all in-container, all fixed)
1. torch 2.10 lane re-lock: libabseil strict minors — torch 2.10
   flatbuffers needs abseil 20260107.1, pyarrow 25.0.0 stack needs
   20260526+. Fixed: release_lane renders per-lane pixi pyarrow pin
   (`_LANE_PIXI_PYARROW = {"2.10": ">=24,<25"}`, pyarrow 24.x on old
   abseil). Commit 28c43b7.
2. `gpu-env-check` pixi task nested-quote bug — every CUDA leg died with
   "unexpected EOF while looking for matching quote" (smoke legs were
   CPU-only so the path had never run). Python strings now escaped double
   quotes. Commit 44c2257.
3. torch 2.12 ships cu128? No — 2.12+ wheels are cu129 only. All five
   2.12-cu128 legs failed at pip. Lane 2.12 now defaults cu129 (cp310–
   cp314 verified on whl/cu129). Commit 47e67c5.
4. VOS publish: astroai/base installs vos under
   `/opt/astroai/venv/cadc/bin` but not on the container PATH; new
   publish script searched standard image locations before refusing to
   pip-install into $HOME. Commit 32fe27c.
5. Launcher race: editing `launch_canfar_gpu_bench.sh` mid-poll corrupted
   running pollers (bash parses script files progressively) — 28/30
   pollers died at post-loop with "syntax error near unexpected token fi"
   (reproduced locally). Daemon now execs a snapshot copy, and exports
   TORCHFITS_ROOT_DIR (snapshot re-run used to double
   benchmarks_results/benchmarks_results/...). Commits 58835da, 32fe27c.
6. Home pollution: no pip installs into $HOME anywhere; all scripts use
   `pixi run python` (never bare python3); `PYTHONNOUSERSITE=1` in
   in-container scripts. Commit 11ee478.

## Aggregate deficits (68 distinct case/op flagged)
CPU (20 legs):
- compressed_hcompress_1:read_full — 78 flags, mean lag 1.022 (~2%)
  (dominant systematic; ~every leg)
- narrow_1000000::read_full — 34 flags, lag 1.097 (10%)
- narrow_100000::read_full — 22 flags, lag 1.163 (16%, worst CPU case)
- compressed_rice_1:read_full — 21 flags, lag 1.007 (borderline noise)
- varlen_100000 read_full/projection — 15 flags, lag 1.06
- large_int16/int8/uint16/medium sporadic — lag 1.04–1.15

CUDA (21 legs):
- compressed_hcompress_1 read_full (+ read_full_gpu) — 168 flags total,
  lag 1.02–1.03
- tiny/small *_1d/2d/3d read_full_gpu — 200+ flags, lag 1.02–1.09
  (GPU transfer/launch overhead dominating small payloads)
- narrow read_full — lag 1.07–1.08
- varlen_100000 read_full/projection — lag 1.22 (22%; worse on GPU)
- narrow_1000000::predicate_filter_selective — 1.574 one-off (57%)

## Cross-version consistency
CPU rows identical across lanes (3,057/leg); no torch-version-dependent
deficits observed — measurement noise is stable, deficits are structural.

## Next (post-benchmark fix plan, report-first)
All deficits map to deep-review findings in
`.cursor/harness/deep-review-1-faithfulness.md` (F1–F4, G1–G2):
- compressed_hcompress read_full: GIL-hold in table_bindings.cpp:205-231
- narrow/varlen: per-batch reopen in _read_scan.py:60-94
- NIOBUF/MINDIRECT silent no-op in CMakeLists
- CUDA small-transfer path: scale-on-device dead path
