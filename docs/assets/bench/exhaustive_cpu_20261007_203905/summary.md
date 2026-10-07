# Benchmark Summary

- Run ID: `exhaustive_cpu_20261007_203905`
- Scopes: `fits, fitstable`
- Total normalized rows: `3057`
- TorchFits deficit rows (all lags): `20`
- TorchFits significant deficits: `11`
- Hostname: `torchfits-gpu-exhaustive-cpu-20261007-203905`
- CPU count: `192`
- torch.get_num_threads(): `8`
- Peak RSS (median across timed rows): `299.9 MB` (max `701.9 MB`)

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
| fitstable | smart | 178/184 | 96.7% | 0 |
| fitstable | specialized | 211/216 | 97.7% | 0 |

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
| fitstable | specialized | 83/84 | 98.8% |

Large-N deficits detected:

| Case | n_points | Lag (x) | Behind (%) |
|---|---:|---:|---:|
| narrow_100000 [read_full] | 100000 | 1.223 | 22.32 |
| narrow_1000000 [read_full] | 1000000 | 1.209 | 20.94 |
| narrow_1000000 [read_full] | 1000000 | 1.078 | 7.80 |

### Small-N Visible Deficits

No small-N visible deficits detected.

## TorchFits Deficits (Not First)

### FITS - smart

| Case | Operation | TorchFits (s) | Winner | Winner (s) | Process peak RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| compressed_hcompress_1 [read_full] | read_full | 0.026420 | fitsio:fitsio_torch | 0.025684 | 311.9 | 1.029 | 2.86 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |
| compressed_hcompress_1 [read_full] | read_full | 0.046332 | fitsio:fitsio_torch | 0.045678 | 295.5 | 1.014 | 1.43 | on | torchfits-gpu-exhaustive-cpu-20261007-203905 |

### FITS - specialized

| Case | Operation | TorchFits (s) | Winner | Winner (s) | Process peak RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| compressed_hcompress_1 [read_full] | read_full | 0.026298 | fitsio:fitsio_torch | 0.025684 | 311.9 | 1.024 | 2.39 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |
| compressed_hcompress_1 [read_full] | read_full | 0.046235 | fitsio:fitsio_torch | 0.045678 | 295.5 | 1.012 | 1.22 | on | torchfits-gpu-exhaustive-cpu-20261007-203905 |

### FITSTABLE - smart

| Case | Operation | TorchFits (s) | Winner | Winner (s) | Process peak RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| narrow_100000 [read_full] | read_full | 0.000724 | fitsio:fitsio_torch | 0.000592 | 382.8 | 1.223 | 22.32 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |
| narrow_1000000 [read_full] | read_full | 0.006218 | fitsio:fitsio_torch | 0.005141 | 444.8 | 1.209 | 20.94 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |
| ascii_10000 [predicate_filter] | predicate_filter | 0.000427 | fitsio:fitsio_torch | 0.000383 | 509.3 | 1.116 | 11.61 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |
| ascii_10000 [predicate_filter_selective] | predicate_filter_selective | 0.000423 | fitsio:fitsio_torch | 0.000382 | 509.3 | 1.106 | 10.62 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |
| ascii_1000 [predicate_filter_selective] | predicate_filter_selective | 0.000181 | fitsio:fitsio_torch | 0.000172 | 509.3 | 1.054 | 5.36 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |
| ascii_1000 [predicate_filter] | predicate_filter | 0.000182 | fitsio:fitsio_torch | 0.000173 | 509.3 | 1.050 | 5.00 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |

### FITSTABLE - specialized

| Case | Operation | TorchFits (s) | Winner | Winner (s) | Process peak RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| ascii_10000 [predicate_filter] | predicate_filter | 0.000430 | fitsio:fitsio | 0.000373 | 509.3 | 1.153 | 15.29 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |
| ascii_10000 [predicate_filter_selective] | predicate_filter_selective | 0.000428 | fitsio:fitsio | 0.000372 | 509.3 | 1.149 | 14.91 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |
| ascii_1000 [predicate_filter_selective] | predicate_filter_selective | 0.000185 | fitsio:fitsio | 0.000165 | 509.3 | 1.120 | 11.96 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |
| ascii_1000 [predicate_filter] | predicate_filter | 0.000185 | fitsio:fitsio | 0.000165 | 509.3 | 1.119 | 11.86 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |
| narrow_1000000 [read_full] | read_full | 0.006154 | fitsio:fitsio | 0.005709 | 444.8 | 1.078 | 7.80 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |
| narrow_1000 [predicate_filter_selective] | predicate_filter_selective | 0.000176 | fitsio:fitsio | 0.000168 | 381.0 | 1.047 | 4.70 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |
| mixed_1000 [predicate_filter_selective] | predicate_filter_selective | 0.000190 | fitsio:fitsio | 0.000185 | 381.3 | 1.028 | 2.82 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |
| narrow_1000 [predicate_filter] | predicate_filter | 0.000174 | fitsio:fitsio | 0.000169 | 379.7 | 1.028 | 2.77 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |
| varlen_1000 [predicate_filter] | predicate_filter | 0.000172 | fitsio:fitsio | 0.000168 | 509.2 | 1.022 | 2.19 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |
| narrow_1000000 [predicate_filter_selective] | predicate_filter_selective | 0.005311 | astropy:astropy | 0.005272 | 444.8 | 1.007 | 0.73 | off | torchfits-gpu-exhaustive-cpu-20261007-203905 |

## Notes

- Strict mmap fairness is enforced in comparable sets. Rows with unmatched mmap controls are marked `SKIPPED`.
- `Process peak RSS` is the interpreter process's peak resident set, sampled by `bench_timing._RssPeakSampler`. It is the same for every method in a comparison group and is **not** a per-library memory figure; ranking is time-based.
- Rankings are family-specific and never mix smart vs specialized method families.
