---
name: science-core
description: Cross-cutting scientific discipline for the astroai/sfabbro astronomy stack — scientific correctness, math rigor, statistical principles, data-driven validation, science impact
metadata:
  domains: astronomy, astrophysics, machine-learning
---

# Science core discipline

Applies to every repo in the ecosystem (torchsky, torchfits, uspm, cosmodist,
xmatch, cfhtcast, weightmask, torchregress, torchz, zensus). Read the repo's
`AGENTS.md` for repo-specific rules; this skill is the shared scientific
standard. When repo guidance conflicts, `AGENTS.md` wins.

## Scientific correctness

- Exact, well-founded methods over heuristics. No ad hoc thresholds or
  data-dependent branching without a numerical, algorithmic, or benchmark
  justification.
- Verify against trusted baselines: upstream libraries (astropy, healpy,
  reproject, CFITSIO), analytic ground truth, or closed-form solutions.
- State units, coordinate frames, and physical conventions explicitly at
  boundaries; never mix radians/degrees or frames silently.
- Fail closed on invalid states; assert shape/dtype invariants at function
  boundaries.

## Math logic

- Derive before coding: write the math (loss, Jacobian, propagation, adjoint)
  down and check it against a reference before implementing.
- Precision: float64 for astronomy numerics unless measured; preserve
  differentiability and device behavior in public torch paths.
- Numerical stability: log-space where needed, stable softplus/logsumexp,
  avoid catastrophic cancellation; test edge regimes (zero variance, empty
  bins, NaN handling).
- When adding a fast path, keep a correct reference path and a parity test.

## Statistics-principled

- Always quantify uncertainty: error bars, predictive distributions, or
  coverage — never bare point estimates for science claims.
- Calibration over confidence: validate that stated coverage matches
  empirical coverage (conformal, quantile, or histogram checks).
- Use proper scoring rules and honest metrics (CRPS, interval score, NLL,
  coverage), not just MSE.
- Document priors, weights, and missing-data handling; make assumptions
  explicit and testable.
- No p-hacking or cherry-picking: pre-register comparisons, report all
  trials, and state significance honestly.

## Data-driven

- Validate on real survey data and realistic simulations, not toy examples
  alone. Real-data or upstream-fixture checks are required for user-facing
  behavior.
- Benchmarks on real workloads with honest methodology: fixed sizes, exact
  parity, throughput AND memory, committed baseline gates.
- Data are sacred in this stack: keep data in native frames where the design
  says so; never silently resample or transform for convenience.

## Science impact

- Prioritize work by expected scientific and operational impact, then
  correctness risk, then performance at survey scale.
- Reproducibility is non-negotiable: every experiment traceable (session
  logs, configs, seeds, versions), results archivable, claims backed by
  evidence.
- Deliver publishable, skeptical-audience-grade results: interpretable
  models, documented physical/statistical motivations, honest comparisons
  against the state of the art (e.g. SED fitting for photo-z).
