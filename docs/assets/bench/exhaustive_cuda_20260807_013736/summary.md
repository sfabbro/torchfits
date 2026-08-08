# Benchmark Summary

- Run ID: `exhaustive_cuda_20260807_013736`
- Scopes: `fits, fitstable`
- Total normalized rows: `4315`
- TorchFits deficit rows (all lags): `26`
- TorchFits significant deficits: `1`
- Hostname: `torchfits-gpu-exhaustive-cuda-20260807-013736`
- CPU count: `96`
- torch.get_num_threads(): `8`
- Peak RSS (median across timed rows): `719.3 MB` (max `953.5 MB`)

## Domain Coverage

| Domain | Rows | Skipped |
|---|---:|---:|
| fits | 2943 | 85 |
| fitstable | 1372 | 104 |

## Astronomer Scorecard

| Domain | Family | TorchFits First | Win Rate | Legacy In Ranking |
|---|---|---:|---:|---:|
| fits | smart | 334/334 | 100.0% | 0 |
| fits | specialized | 419/419 | 100.0% | 0 |
| fitstable | smart | 183/184 | 99.5% | 0 |
| fitstable | specialized | 216/216 | 100.0% | 0 |

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
| narrow_1000000 [read_full] | 1000000 | 1.083 | 8.30 |

### Small-N Visible Deficits

No small-N visible deficits detected.

## TorchFits Deficits (Not First)

### FITS - smart

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| tiny_int8_1d [read_full @ cuda] | read_full | 0.000118 | 766.8 | fitsio:fitsio_torch_device | 0.000108 | 766.8 | 1.096 | 9.60 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| tiny_float64_3d [read_full @ cuda] | read_full | 0.000131 | 766.8 | fitsio:fitsio_torch_device | 0.000121 | 766.8 | 1.079 | 7.90 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| tiny_int16_2d [read_full @ cuda] | read_full | 0.000117 | 766.8 | fitsio:fitsio_torch_device | 0.000111 | 766.8 | 1.059 | 5.95 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| tiny_int64_1d [read_full @ cuda] | read_full | 0.000111 | 766.8 | fitsio:fitsio_torch_device | 0.000106 | 766.8 | 1.048 | 4.76 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| compressed_hcompress_1 [read_full @ cuda] | read_full | 0.030717 | 729.4 | fitsio:fitsio_torch_device | 0.029629 | 729.4 | 1.037 | 3.67 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| medium_int8_1d [read_full @ cuda] | read_full | 0.000146 | 766.8 | fitsio:fitsio_torch_device | 0.000141 | 766.8 | 1.035 | 3.54 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| compressed_hcompress_1 [read_full @ cuda] | read_full | 0.030599 | 699.1 | fitsio:fitsio_torch_device | 0.029567 | 699.1 | 1.035 | 3.49 | on | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| tiny_int64_2d [read_full @ cuda] | read_full | 0.000126 | 766.8 | fitsio:fitsio_torch_device | 0.000122 | 766.8 | 1.033 | 3.27 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| tiny_int8_2d [read_full @ cuda] | read_full | 0.000118 | 766.8 | fitsio:fitsio_torch_device | 0.000114 | 766.8 | 1.029 | 2.85 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| compressed_hcompress_1 [read_full] | read_full | 0.030266 | 606.7 | fitsio:fitsio_torch | 0.029493 | 606.7 | 1.026 | 2.62 | on | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| tiny_float64_1d [read_full @ cuda] | read_full | 0.000113 | 766.8 | fitsio:fitsio_torch_device | 0.000110 | 766.8 | 1.024 | 2.41 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| small_int8_1d [read_full @ cuda] | read_full | 0.000113 | 766.8 | fitsio:fitsio_torch_device | 0.000111 | 766.8 | 1.024 | 2.35 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| compressed_hcompress_1 [read_full] | read_full | 0.030242 | 728.3 | fitsio:fitsio_torch | 0.029620 | 728.3 | 1.021 | 2.10 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| small_uint16_2d [read_full @ cuda] | read_full | 0.000153 | 766.8 | fitsio:fitsio_torch_device | 0.000151 | 766.8 | 1.019 | 1.92 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| tiny_float64_2d [read_full @ cuda] | read_full | 0.000119 | 766.8 | fitsio:fitsio_torch_device | 0.000117 | 766.8 | 1.014 | 1.41 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| small_int64_1d [read_full @ cuda] | read_full | 0.000133 | 766.8 | fitsio:fitsio_torch_device | 0.000131 | 766.8 | 1.014 | 1.37 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| medium_int8_2d [read_full] | read_full | 0.000321 | 765.8 | fitsio:fitsio_torch | 0.000317 | 765.8 | 1.014 | 1.36 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| tiny_int32_2d [read_full @ cuda] | read_full | 0.000114 | 766.8 | fitsio:fitsio_torch_device | 0.000114 | 766.8 | 1.007 | 0.68 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |

### FITS - specialized

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| compressed_hcompress_1 [read_full @ cuda] | read_full | 0.030734 | 729.4 | fitsio:fitsio_torch_device_specialized | 0.029686 | 729.4 | 1.035 | 3.53 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| compressed_hcompress_1 [read_full @ cuda] | read_full | 0.030575 | 699.1 | fitsio:fitsio_torch_device_specialized | 0.029556 | 699.1 | 1.034 | 3.45 | on | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| compressed_hcompress_1 [read_full] | read_full | 0.030308 | 606.7 | fitsio:fitsio_torch | 0.029493 | 606.7 | 1.028 | 2.76 | on | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| compressed_hcompress_1 [read_full] | read_full | 0.030328 | 728.3 | fitsio:fitsio_torch | 0.029620 | 728.3 | 1.024 | 2.39 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| small_int16_3d [read_full] | read_full | 0.000156 | 765.8 | fitsio:fitsio_torch | 0.000154 | 765.8 | 1.016 | 1.64 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |

### FITSTABLE - smart

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| narrow_1000000 [read_full] | read_full | 0.007678 | 715.5 | fitsio:fitsio_torch | 0.007090 | 715.5 | 1.083 | 8.30 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |
| typed_100000 [predicate_filter] | predicate_filter | 0.002087 | 738.1 | fitsio:fitsio_torch | 0.002022 | 738.1 | 1.032 | 3.23 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |

### FITSTABLE - specialized

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| typed_100000 [predicate_filter] | predicate_filter | 0.002103 | 738.1 | fitsio:fitsio | 0.002037 | 738.1 | 1.033 | 3.26 | off | torchfits-gpu-exhaustive-cuda-20260807-013736 |

## Notes

- Strict mmap fairness is enforced in comparable sets. Rows with unmatched mmap controls are marked `SKIPPED`.
- Rankings are family-specific and never mix smart vs specialized method families.
