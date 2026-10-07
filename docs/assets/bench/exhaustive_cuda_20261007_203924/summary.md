# Benchmark Summary

- Run ID: `exhaustive_cuda_20261007_203924`
- Scopes: `fits, fitstable`
- Total normalized rows: `4315`
- TorchFits deficit rows (all lags): `44`
- TorchFits significant deficits: `7`
- Hostname: `torchfits-gpu-exhaustive-cuda-20261007-203924`
- CPU count: `96`
- torch.get_num_threads(): `8`
- Peak RSS (median across timed rows): `762.9 MB` (max `1101.6 MB`)

## Domain Coverage

| Domain | Rows | Skipped |
|---|---:|---:|
| fits | 2943 | 85 |
| fitstable | 1372 | 104 |

## Astronomer Scorecard

| Domain | Family | TorchFits First | Win Rate | Legacy In Ranking |
|---|---|---:|---:|---:|
| fits | smart | 333/334 | 99.7% | 0 |
| fits | specialized | 418/419 | 99.8% | 0 |
| fitstable | smart | 182/184 | 98.9% | 0 |
| fitstable | specialized | 213/216 | 98.6% | 0 |

- TorchFits devices observed in this run: `cpu, cuda`
- Smart-family tables are the primary adoption view for astronomers (performance + portability).

## Adoption Checks

- `large-N` threshold: `n_points >= 100000`
- `small-N perceived` threshold: `torchfits_time_s < 0.000500s`
- `small-N max lag` threshold: `lag_ratio < 10.0x`

### Large-N Leadership

| Domain | Family | TorchFits First (large-N) | Win Rate |
|---|---|---:|---:|
| fitstable | smart | 69/70 | 98.6% |
| fitstable | specialized | 84/84 | 100.0% |

Large-N deficits detected:

| Case | n_points | Lag (x) | Behind (%) |
|---|---:|---:|---:|
| narrow_100000 [read_full] | 100000 | 1.129 | 12.92 |

### Small-N Visible Deficits

| Case | TorchFits (s) | Lag (x) | Behind (%) | Impact |
|---|---:|---:|---:|---|
| scaled_large [read_full @ cuda] | 0.007276 | 1.267 | 26.72 | visible |
| scaled_large [read_full @ cuda] | 0.007262 | 1.259 | 25.88 | visible |
| ascii_10000 [predicate_filter] | 0.000555 | 1.051 | 5.09 | visible |
| ascii_10000 [predicate_filter] | 0.000555 | 1.100 | 9.99 | visible |
| ascii_10000 [predicate_filter_selective] | 0.000560 | 1.081 | 8.09 | visible |

## TorchFits Deficits (Not First)

### FITS - smart

| Case | Operation | TorchFits (s) | Winner | Winner (s) | Process peak RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| scaled_large [read_full @ cuda] | read_full | 0.007276 | fitsio:fitsio_torch_device | 0.005742 | 772.7 | 1.267 | 26.72 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| medium_int8_3d [read_full @ cuda] | read_full | 0.000642 | fitsio:fitsio_torch_device | 0.000522 | 772.7 | 1.231 | 23.14 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| large_int8_2d [read_full @ cuda] | read_full | 0.001182 | fitsio:fitsio_torch_device | 0.001042 | 772.7 | 1.134 | 13.41 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| tiny_int8_1d [read_full @ cuda] | read_full | 0.000107 | fitsio:fitsio_torch_device | 0.000095 | 772.7 | 1.134 | 13.35 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| tiny_int64_1d [read_full @ cuda] | read_full | 0.000107 | fitsio:fitsio_torch_device | 0.000096 | 772.7 | 1.118 | 11.82 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| tiny_int16_2d [read_full @ cuda] | read_full | 0.000109 | fitsio:fitsio_torch_device | 0.000098 | 772.7 | 1.112 | 11.16 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| small_int16_1d [read_full @ cuda] | read_full | 0.000112 | fitsio:fitsio_torch_device | 0.000102 | 772.7 | 1.104 | 10.35 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| tiny_int32_2d [read_full @ cuda] | read_full | 0.000112 | fitsio:fitsio_torch_device | 0.000102 | 772.7 | 1.091 | 9.15 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| tiny_int8_3d [read_full @ cuda] | read_full | 0.000107 | fitsio:fitsio_torch_device | 0.000099 | 772.7 | 1.073 | 7.33 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| scaled_small [read_full @ cuda] | read_full | 0.000203 | fitsio:fitsio_torch_device | 0.000190 | 772.7 | 1.072 | 7.21 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| scaled_medium [read_full @ cuda] | read_full | 0.001335 | fitsio:fitsio_torch_device | 0.001247 | 772.7 | 1.070 | 7.00 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| tiny_int32_3d [read_full @ cuda] | read_full | 0.000109 | fitsio:fitsio_torch_device | 0.000102 | 772.7 | 1.067 | 6.73 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| medium_int8_1d [read_full @ cuda] | read_full | 0.000128 | fitsio:fitsio_torch_device | 0.000120 | 772.7 | 1.067 | 6.67 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| small_int8_2d [read_full @ cuda] | read_full | 0.000144 | fitsio:fitsio_torch_device | 0.000136 | 772.7 | 1.061 | 6.05 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| tiny_float32_1d [read_full @ cuda] | read_full | 0.000103 | fitsio:fitsio_torch_device | 0.000097 | 772.7 | 1.057 | 5.65 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| tiny_int16_3d [read_full @ cuda] | read_full | 0.000110 | fitsio:fitsio_torch_device | 0.000105 | 772.7 | 1.051 | 5.06 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| compressed_hcompress_1 [read_full @ cuda] | read_full | 0.030648 | fitsio:fitsio_torch_device | 0.029458 | 772.7 | 1.040 | 4.04 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| tiny_float64_2d [read_full @ cuda] | read_full | 0.000113 | fitsio:fitsio_torch_device | 0.000109 | 772.7 | 1.038 | 3.82 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| small_uint16_2d [read_full @ cuda] | read_full | 0.000147 | fitsio:fitsio_torch_device | 0.000141 | 772.7 | 1.037 | 3.71 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| small_int16_2d [read_full @ cuda] | read_full | 0.000138 | fitsio:fitsio_torch_device | 0.000133 | 772.7 | 1.036 | 3.60 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| compressed_hcompress_1 [read_full @ cuda] | read_full | 0.030549 | fitsio:fitsio_torch_device | 0.029493 | 719.3 | 1.036 | 3.58 | on | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| compressed_hcompress_1 [read_full] | read_full | 0.030408 | fitsio:fitsio_torch | 0.029497 | 730.9 | 1.031 | 3.09 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| compressed_hcompress_1 [read_full] | read_full | 0.030547 | fitsio:fitsio_torch | 0.029878 | 693.2 | 1.022 | 2.24 | on | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| tiny_float32_3d [read_full @ cuda] | read_full | 0.000105 | fitsio:fitsio_torch_device | 0.000104 | 772.7 | 1.014 | 1.44 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| small_int8_1d [read_full @ cuda] | read_full | 0.000125 | fitsio:fitsio_torch_device | 0.000123 | 772.7 | 1.012 | 1.19 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| tiny_float32_2d [read_full @ cuda] | read_full | 0.000106 | fitsio:fitsio_torch_device | 0.000105 | 772.7 | 1.007 | 0.68 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| tiny_int8_2d [read_full @ cuda] | read_full | 0.000104 | fitsio:fitsio_torch_device | 0.000103 | 772.7 | 1.005 | 0.51 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| small_int16_3d [read_full @ cuda] | read_full | 0.000181 | fitsio:fitsio_torch_device | 0.000180 | 772.7 | 1.004 | 0.43 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| small_float32_1d [read_full @ cuda] | read_full | 0.000109 | fitsio:fitsio_torch_device | 0.000108 | 772.7 | 1.003 | 0.32 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |

### FITS - specialized

| Case | Operation | TorchFits (s) | Winner | Winner (s) | Process peak RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| scaled_large [read_full @ cuda] | read_full | 0.007262 | fitsio:fitsio_torch_device_specialized | 0.005769 | 772.7 | 1.259 | 25.88 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| large_int8_2d [read_full @ cuda] | read_full | 0.001195 | fitsio:fitsio_torch_device_specialized | 0.001034 | 772.7 | 1.156 | 15.57 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| compressed_hcompress_1 [read_full @ cuda] | read_full | 0.030566 | fitsio:fitsio_torch_device_specialized | 0.029398 | 772.7 | 1.040 | 3.97 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| compressed_hcompress_1 [read_full @ cuda] | read_full | 0.030623 | fitsio:fitsio_torch_device_specialized | 0.029524 | 719.3 | 1.037 | 3.72 | on | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| compressed_hcompress_1 [read_full] | read_full | 0.030650 | fitsio:fitsio_torch | 0.029878 | 693.2 | 1.026 | 2.58 | on | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| compressed_hcompress_1 [read_full] | read_full | 0.030201 | fitsio:fitsio_torch | 0.029497 | 730.9 | 1.024 | 2.38 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |

### FITSTABLE - smart

| Case | Operation | TorchFits (s) | Winner | Winner (s) | Process peak RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| narrow_100000 [read_full] | read_full | 0.000836 | fitsio:fitsio_torch | 0.000740 | 790.4 | 1.129 | 12.92 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| ascii_10000 [predicate_filter] | predicate_filter | 0.000555 | fitsio:fitsio_torch | 0.000528 | 911.0 | 1.051 | 5.09 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| ascii_10000 [predicate_filter_selective] | predicate_filter_selective | 0.000549 | fitsio:fitsio_torch | 0.000533 | 911.0 | 1.030 | 3.00 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| ascii_1000 [predicate_filter] | predicate_filter | 0.000237 | fitsio:fitsio_torch | 0.000236 | 911.0 | 1.005 | 0.50 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |

### FITSTABLE - specialized

| Case | Operation | TorchFits (s) | Winner | Winner (s) | Process peak RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| ascii_10000 [predicate_filter] | predicate_filter | 0.000555 | fitsio:fitsio | 0.000505 | 911.0 | 1.100 | 9.99 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| ascii_10000 [predicate_filter_selective] | predicate_filter_selective | 0.000560 | fitsio:fitsio | 0.000518 | 911.0 | 1.081 | 8.09 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| ascii_1000 [predicate_filter] | predicate_filter | 0.000237 | fitsio:fitsio | 0.000224 | 911.0 | 1.060 | 5.99 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| ascii_1000 [predicate_filter_selective] | predicate_filter_selective | 0.000234 | fitsio:fitsio | 0.000224 | 911.0 | 1.042 | 4.22 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |
| narrow_1000 [predicate_filter_selective] | predicate_filter_selective | 0.000236 | fitsio:fitsio | 0.000232 | 773.7 | 1.015 | 1.51 | off | torchfits-gpu-exhaustive-cuda-20261007-203924 |

## Notes

- Strict mmap fairness is enforced in comparable sets. Rows with unmatched mmap controls are marked `SKIPPED`.
- `Process peak RSS` is the interpreter process's peak resident set, sampled by `bench_timing._RssPeakSampler`. It is the same for every method in a comparison group and is **not** a per-library memory figure; ranking is time-based.
- Rankings are family-specific and never mix smart vs specialized method families.
