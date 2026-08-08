# Benchmark Summary

- Run ID: `exhaustive_cpu_20260807_013736`
- Scopes: `fits, fitstable`
- Total normalized rows: `3057`
- TorchFits deficit rows (all lags): `11`
- TorchFits significant deficits: `3`
- Hostname: `torchfits-gpu-exhaustive-cpu-20260807-013736`
- CPU count: `192`
- torch.get_num_threads(): `8`
- Peak RSS (median across timed rows): `293.8 MB` (max `630.5 MB`)

## Domain Coverage

| Domain | Rows | Skipped |
|---|---:|---:|
| fits | 1689 | 85 |
| fitstable | 1368 | 104 |

## Astronomer Scorecard

| Domain | Family | TorchFits First | Win Rate | Legacy In Ranking |
|---|---|---:|---:|---:|
| fits | smart | 157/157 | 100.0% | 0 |
| fits | specialized | 242/242 | 100.0% | 0 |
| fitstable | smart | 182/184 | 98.9% | 0 |
| fitstable | specialized | 215/216 | 99.5% | 0 |

- TorchFits devices observed in this run: `-`
- Smart-family tables are the primary adoption view for astronomers (performance + portability).

## Adoption Checks

- `large-N` threshold: `n_points >= 100000`
- `small-N perceived` threshold: `torchfits_time_s < 0.000500s`
- `small-N max lag` threshold: `lag_ratio < 10.0x`

### Large-N Leadership

| Domain | Family | TorchFits First (large-N) | Win Rate |
|---|---|---:|---:|
| fitstable | smart | 68/70 | 97.1% |
| fitstable | specialized | 84/84 | 100.0% |

Large-N deficits detected:

| Case | n_points | Lag (x) | Behind (%) |
|---|---:|---:|---:|
| narrow_100000 [read_full] | 100000 | 1.365 | 36.46 |
| narrow_1000000 [read_full] | 1000000 | 1.210 | 20.96 |

### Small-N Visible Deficits

No small-N visible deficits detected.

## TorchFits Deficits (Not First)

### FITS - smart

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| compressed_hcompress_1 [read_full] | read_full | 0.045697 | 293.8 | fitsio:fitsio_torch | 0.044738 | 293.8 | 1.021 | 2.14 | on | torchfits-gpu-exhaustive-cpu-20260807-013736 |
| compressed_hcompress_1 [read_full] | read_full | 0.045559 | 309.3 | fitsio:fitsio_torch | 0.044618 | 309.3 | 1.021 | 2.11 | off | torchfits-gpu-exhaustive-cpu-20260807-013736 |

### FITS - specialized

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| compressed_hcompress_1 [read_full] | read_full | 0.045719 | 293.8 | fitsio:fitsio_torch | 0.044738 | 293.8 | 1.022 | 2.19 | on | torchfits-gpu-exhaustive-cpu-20260807-013736 |
| compressed_hcompress_1 [read_full] | read_full | 0.045540 | 309.3 | fitsio:fitsio_torch | 0.044618 | 309.3 | 1.021 | 2.07 | off | torchfits-gpu-exhaustive-cpu-20260807-013736 |

### FITSTABLE - smart

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| narrow_100000 [read_full] | read_full | 0.001220 | 380.4 | fitsio:fitsio_torch | 0.000894 | 380.4 | 1.365 | 36.46 | off | torchfits-gpu-exhaustive-cpu-20260807-013736 |
| narrow_1000000 [read_full] | read_full | 0.006000 | 399.4 | fitsio:fitsio_torch | 0.004961 | 399.4 | 1.210 | 20.96 | off | torchfits-gpu-exhaustive-cpu-20260807-013736 |
| ascii_10000 [predicate_filter] | predicate_filter | 0.000383 | 422.3 | fitsio:fitsio_torch | 0.000374 | 422.3 | 1.023 | 2.27 | off | torchfits-gpu-exhaustive-cpu-20260807-013736 |

### FITSTABLE - specialized

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| ascii_10000 [predicate_filter] | predicate_filter | 0.000380 | 422.3 | fitsio:fitsio | 0.000357 | 422.3 | 1.064 | 6.40 | off | torchfits-gpu-exhaustive-cpu-20260807-013736 |
| ascii_10000 [predicate_filter_selective] | predicate_filter_selective | 0.000372 | 422.3 | fitsio:fitsio | 0.000360 | 422.3 | 1.034 | 3.35 | off | torchfits-gpu-exhaustive-cpu-20260807-013736 |
| narrow_100000 [read_full] | read_full | 0.001127 | 380.4 | fitsio:fitsio | 0.001111 | 380.4 | 1.014 | 1.44 | off | torchfits-gpu-exhaustive-cpu-20260807-013736 |
| narrow_1000000 [read_full] | read_full | 0.005927 | 403.5 | fitsio:fitsio | 0.005855 | 403.5 | 1.012 | 1.24 | off | torchfits-gpu-exhaustive-cpu-20260807-013736 |

## Notes

- Strict mmap fairness is enforced in comparable sets. Rows with unmatched mmap controls are marked `SKIPPED`.
- Rankings are family-specific and never mix smart vs specialized method families.
