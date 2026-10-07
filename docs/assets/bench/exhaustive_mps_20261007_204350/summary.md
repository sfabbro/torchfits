# Benchmark Summary

- Run ID: `exhaustive_mps_20261007_204350`
- Scopes: `fits, fitstable`
- Total normalized rows: `4227`
- TorchFits deficit rows (all lags): `97`
- TorchFits significant deficits: `58`
- Hostname: `NRC-054711`
- CPU count: `8`
- torch.get_num_threads(): `4`
- Peak RSS (median across timed rows): `226.5 MB` (max `838.1 MB`)

## Domain Coverage

| Domain | Rows | Skipped |
|---|---:|---:|
| fits | 2855 | 85 |
| fitstable | 1372 | 104 |

## Astronomer Scorecard

| Domain | Family | TorchFits First | Win Rate | Legacy In Ranking |
|---|---|---:|---:|---:|
| fits | smart | 292/312 | 93.6% | 0 |
| fits | specialized | 374/397 | 94.2% | 0 |
| fitstable | smart | 180/184 | 97.8% | 0 |
| fitstable | specialized | 205/216 | 94.9% | 0 |

- TorchFits devices observed in this run: `cpu, mps`
- Smart-family tables are the primary adoption view for astronomers (performance + portability).

## Adoption Checks

- `large-N` threshold: `n_points >= 100000`
- `small-N perceived` threshold: `torchfits_time_s < 0.000500s`
- `small-N max lag` threshold: `lag_ratio < 10.0x`

### Large-N Leadership

| Domain | Family | TorchFits First (large-N) | Win Rate |
|---|---|---:|---:|
| fitstable | smart | 67/70 | 95.7% |
| fitstable | specialized | 75/84 | 89.3% |

Large-N deficits detected:

| Case | n_points | Lag (x) | Behind (%) |
|---|---:|---:|---:|
| varlen_100000 [predicate_filter_selective] | 100000 | 1.618 | 61.85 |
| typed_100000 [predicate_filter] | 100000 | 1.555 | 55.50 |
| wide_100000 [predicate_filter] | 100000 | 1.128 | 12.84 |
| narrow_100000 [predicate_filter_selective] | 100000 | 2.249 | 124.93 |
| narrow_1000000 [predicate_filter_selective] | 1000000 | 1.621 | 62.10 |
| mixed_1000000 [predicate_filter_selective] | 1000000 | 1.505 | 50.50 |
| typed_100000 [predicate_filter_selective] | 100000 | 1.431 | 43.11 |
| typed_100000 [predicate_filter] | 100000 | 1.406 | 40.63 |
| mixed_1000000 [predicate_filter] | 1000000 | 1.162 | 16.16 |
| narrow_1000000 [predicate_filter] | 1000000 | 1.150 | 15.01 |
| wide_100000 [predicate_filter_selective] | 100000 | 1.150 | 14.96 |
| varlen_100000 [predicate_filter_selective] | 100000 | 1.086 | 8.58 |

### Small-N Visible Deficits

| Case | TorchFits (s) | Lag (x) | Behind (%) | Impact |
|---|---:|---:|---:|---|
| small_int64_1d [read_full] | 0.000676 | 2.439 | 143.87 | visible |
| small_int32_2d [read_full] | 0.001010 | 2.171 | 117.10 | visible |
| small_uint16_2d [read_full @ mps] | 0.001244 | 1.971 | 97.06 | visible |
| tiny_int64_3d [read_full] | 0.001169 | 1.713 | 71.25 | visible |
| scaled_large [read_full @ mps] | 0.014319 | 1.701 | 70.07 | visible |
| small_int16_3d [read_full] | 0.000889 | 1.598 | 59.80 | visible |
| small_int64_1d [read_full @ mps] | 0.002944 | 1.483 | 48.25 | visible |
| medium_int8_1d [read_full @ mps] | 0.001328 | 1.481 | 48.09 | visible |
| scaled_medium [read_full @ mps] | 0.004292 | 1.469 | 46.88 | visible |
| small_int16_2d [read_full @ mps] | 0.001159 | 1.464 | 46.43 | visible |
| timeseries_frame_000 [read_full @ mps] | 0.001024 | 1.452 | 45.18 | visible |
| tiny_float32_1d [read_full @ mps] | 0.000899 | 1.392 | 39.19 | visible |
| small_uint32_2d [read_full @ mps] | 0.002266 | 1.294 | 29.38 | visible |
| small_int8_2d [read_full @ mps] | 0.001091 | 1.277 | 27.69 | visible |
| compressed_rice_1 [cutout_100x100] | 0.003951 | 1.193 | 19.32 | visible |
| compressed_rice_1 [read_full @ mps] | 0.022864 | 1.182 | 18.23 | visible |
| large_int16_2d [read_full @ mps] | 0.004724 | 1.144 | 14.43 | visible |
| compressed_rice_1 [read_full @ mps] | 0.022472 | 1.101 | 10.14 | visible |
| large_int64_2d [read_full @ mps] | 0.017513 | 1.081 | 8.14 | visible |
| large_uint16_2d [read_full @ mps] | 0.012263 | 1.045 | 4.48 | visible |
| small_int32_1d [read_full] | 0.000819 | 2.801 | 180.10 | visible |
| small_float32_1d [read_full] | 0.000953 | 2.796 | 179.58 | visible |
| small_int16_2d [read_full] | 0.000811 | 2.304 | 130.44 | visible |
| timeseries_frame_004 [read_full] | 0.000772 | 2.132 | 113.24 | visible |
| small_uint16_2d [read_full @ mps] | 0.001412 | 2.015 | 101.50 | visible |
| small_int64_1d [read_full] | 0.000537 | 1.938 | 93.78 | visible |
| small_float32_1d [read_full @ mps] | 0.001721 | 1.891 | 89.10 | visible |
| timeseries_frame_003 [read_full] | 0.001308 | 1.857 | 85.75 | visible |
| scaled_medium [read_full @ mps] | 0.005024 | 1.736 | 73.63 | visible |
| scaled_large [read_full @ mps] | 0.012783 | 1.507 | 50.69 | visible |
| multi_mef_10ext [cutout_100x100] | 0.001082 | 1.471 | 47.11 | visible |
| tiny_float32_2d [read_full @ mps] | 0.000788 | 1.369 | 36.87 | visible |
| timeseries_frame_002 [read_full @ mps] | 0.001086 | 1.267 | 26.70 | visible |
| medium_int32_2d [read_full @ mps] | 0.003325 | 1.198 | 19.76 | visible |
| large_float32_1d [read_full @ mps] | 0.005873 | 1.195 | 19.51 | visible |
| compressed_rice_1 [cutout_100x100] | 0.003917 | 1.183 | 18.28 | visible |
| medium_uint16_2d [read_full @ mps] | 0.002660 | 1.129 | 12.88 | visible |
| compressed_rice_1 [read_full @ mps] | 0.021945 | 1.112 | 11.15 | visible |
| large_float32_2d [read_full @ mps] | 0.008269 | 1.084 | 8.36 | visible |
| large_float32_1d [read_full @ mps] | 0.004032 | 1.076 | 7.56 | visible |
| medium_float32_3d [read_full @ mps] | 0.004767 | 1.070 | 6.99 | visible |
| compressed_rice_1 [read_full @ mps] | 0.018417 | 1.022 | 2.19 | visible |
| compressed_gzip_2 [read_full @ mps] | 0.043027 | 1.005 | 0.54 | visible |
| narrow_10000 [read_full] | 0.000856 | 1.427 | 42.70 | visible |
| varlen_10000 [predicate_filter_selective] | 0.002217 | 1.327 | 32.65 | visible |
| ascii_10000 [predicate_filter_selective] | 0.001381 | 1.051 | 5.13 | visible |

## TorchFits Deficits (Not First)

### FITS - smart

| Case | Operation | TorchFits (s) | Winner | Winner (s) | Process peak RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| small_int64_1d [read_full] | read_full | 0.000676 | fitsio:fitsio_torch | 0.000277 | 193.2 | 2.439 | 143.87 | off | NRC-054711 |
| small_int32_2d [read_full] | read_full | 0.001010 | fitsio:fitsio_torch | 0.000465 | 185.7 | 2.171 | 117.10 | off | NRC-054711 |
| small_uint16_2d [read_full @ mps] | read_full | 0.001244 | fitsio:fitsio_torch_device | 0.000631 | 197.4 | 1.971 | 97.06 | off | NRC-054711 |
| tiny_int64_3d [read_full] | read_full | 0.001169 | fitsio:fitsio_torch | 0.000683 | 180.8 | 1.713 | 71.25 | off | NRC-054711 |
| scaled_large [read_full @ mps] | read_full | 0.014319 | fitsio:fitsio_torch_device | 0.008419 | 200.2 | 1.701 | 70.07 | off | NRC-054711 |
| small_int16_3d [read_full] | read_full | 0.000889 | fitsio:fitsio_torch | 0.000556 | 189.1 | 1.598 | 59.80 | off | NRC-054711 |
| small_int64_1d [read_full @ mps] | read_full | 0.002944 | astropy:astropy_torch_device | 0.001986 | 210.8 | 1.483 | 48.25 | on | NRC-054711 |
| medium_int8_1d [read_full @ mps] | read_full | 0.001328 | fitsio:fitsio_torch_device | 0.000897 | 294.8 | 1.481 | 48.09 | off | NRC-054711 |
| scaled_medium [read_full @ mps] | read_full | 0.004292 | fitsio:fitsio_torch_device | 0.002922 | 299.5 | 1.469 | 46.88 | off | NRC-054711 |
| small_int16_2d [read_full @ mps] | read_full | 0.001159 | fitsio:fitsio_torch_device | 0.000791 | 213.0 | 1.464 | 46.43 | off | NRC-054711 |
| timeseries_frame_000 [read_full @ mps] | read_full | 0.001024 | fitsio:fitsio_torch_device | 0.000705 | 197.4 | 1.452 | 45.18 | off | NRC-054711 |
| tiny_float32_1d [read_full @ mps] | read_full | 0.000899 | fitsio:fitsio_torch_device | 0.000646 | 143.4 | 1.392 | 39.19 | off | NRC-054711 |
| small_uint32_2d [read_full @ mps] | read_full | 0.002266 | fitsio:fitsio_torch_device | 0.001751 | 197.5 | 1.294 | 29.38 | off | NRC-054711 |
| timeseries_frame_003 [read_full @ mps] | read_full | 0.000763 | fitsio:fitsio_torch_device | 0.000595 | 143.2 | 1.281 | 28.15 | off | NRC-054711 |
| small_int8_2d [read_full @ mps] | read_full | 0.001091 | fitsio:fitsio_torch_device | 0.000854 | 198.3 | 1.277 | 27.69 | off | NRC-054711 |
| small_int32_3d [read_full @ mps] | read_full | 0.000926 | fitsio:fitsio_torch_device | 0.000739 | 188.2 | 1.253 | 25.25 | off | NRC-054711 |
| tiny_int64_1d [read_full @ mps] | read_full | 0.000598 | fitsio:fitsio_torch_device | 0.000498 | 139.8 | 1.200 | 20.03 | off | NRC-054711 |
| compressed_rice_1 [cutout_100x100] | cutout_100x100 | 0.003951 | fitsio:fitsio_torch | 0.003312 | 182.0 | 1.193 | 19.32 | n/a | NRC-054711 |
| tiny_int8_1d [read_full @ mps] | read_full | 0.000664 | fitsio:fitsio_torch_device | 0.000559 | 140.4 | 1.188 | 18.83 | off | NRC-054711 |
| small_int16_1d [read_full @ mps] | read_full | 0.000759 | fitsio:fitsio_torch_device | 0.000642 | 240.6 | 1.183 | 18.25 | off | NRC-054711 |
| compressed_rice_1 [read_full @ mps] | read_full | 0.022864 | fitsio:fitsio_torch_device | 0.019339 | 206.5 | 1.182 | 18.23 | on | NRC-054711 |
| tiny_int32_2d [read_full @ mps] | read_full | 0.000602 | fitsio:fitsio_torch_device | 0.000520 | 139.9 | 1.158 | 15.76 | off | NRC-054711 |
| large_int16_2d [read_full @ mps] | read_full | 0.004724 | fitsio:fitsio_torch_device | 0.004128 | 390.7 | 1.144 | 14.43 | off | NRC-054711 |
| tiny_int32_1d [read_full @ mps] | read_full | 0.000638 | fitsio:fitsio_torch_device | 0.000569 | 139.9 | 1.121 | 12.13 | off | NRC-054711 |
| tiny_float32_2d [read_full @ mps] | read_full | 0.000800 | fitsio:fitsio_torch_device | 0.000722 | 143.4 | 1.109 | 10.90 | off | NRC-054711 |
| tiny_int8_3d [read_full @ mps] | read_full | 0.001476 | astropy:astropy_torch_device | 0.001331 | 140.4 | 1.109 | 10.88 | off | NRC-054711 |
| compressed_rice_1 [read_full @ mps] | read_full | 0.022472 | fitsio:fitsio_torch_device | 0.020403 | 178.4 | 1.101 | 10.14 | off | NRC-054711 |
| large_int64_2d [read_full @ mps] | read_full | 0.017513 | fitsio:fitsio_torch_device | 0.016195 | 196.3 | 1.081 | 8.14 | off | NRC-054711 |
| tiny_int16_2d [read_full @ mps] | read_full | 0.000685 | fitsio:fitsio_torch_device | 0.000653 | 142.6 | 1.049 | 4.89 | off | NRC-054711 |
| large_uint16_2d [read_full @ mps] | read_full | 0.012263 | fitsio:fitsio_torch_device | 0.011737 | 241.2 | 1.045 | 4.48 | off | NRC-054711 |
| small_int32_2d [read_full @ mps] | read_full | 0.000694 | fitsio:fitsio_torch_device | 0.000666 | 188.2 | 1.042 | 4.25 | off | NRC-054711 |
| small_int32_1d [read_full] | read_full | 0.000303 | fitsio:fitsio_torch | 0.000293 | 185.5 | 1.037 | 3.69 | off | NRC-054711 |
| tiny_int64_3d [read_full @ mps] | read_full | 0.000737 | fitsio:fitsio_torch_device | 0.000715 | 140.2 | 1.031 | 3.12 | off | NRC-054711 |
| medium_int32_1d [read_full @ mps] | read_full | 0.000978 | fitsio:fitsio_torch_device | 0.000968 | 221.6 | 1.010 | 1.02 | off | NRC-054711 |
| large_uint32_2d [read_full @ mps] | read_full | 0.016405 | fitsio:fitsio_torch_device | 0.016263 | 275.9 | 1.009 | 0.87 | off | NRC-054711 |
| medium_int32_2d [read_full @ mps] | read_full | 0.002593 | fitsio:fitsio_torch_device | 0.002579 | 222.6 | 1.005 | 0.55 | off | NRC-054711 |
| large_int32_1d [read_full @ mps] | read_full | 0.002929 | fitsio:fitsio_torch_device | 0.002925 | 447.0 | 1.002 | 0.16 | off | NRC-054711 |

### FITS - specialized

| Case | Operation | TorchFits (s) | Winner | Winner (s) | Process peak RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| small_int32_1d [read_full] | read_full | 0.000819 | fitsio:fitsio_torch | 0.000293 | 185.5 | 2.801 | 180.10 | off | NRC-054711 |
| small_float32_1d [read_full] | read_full | 0.000953 | fitsio:fitsio_torch | 0.000341 | 212.2 | 2.796 | 179.58 | off | NRC-054711 |
| small_int16_2d [read_full] | read_full | 0.000811 | fitsio:fitsio_torch | 0.000352 | 206.2 | 2.304 | 130.44 | off | NRC-054711 |
| timeseries_frame_004 [read_full] | read_full | 0.000772 | fitsio:fitsio_torch | 0.000362 | 185.1 | 2.132 | 113.24 | off | NRC-054711 |
| small_uint16_2d [read_full @ mps] | read_full | 0.001412 | fitsio:fitsio_torch_device_specialized | 0.000701 | 197.4 | 2.015 | 101.50 | off | NRC-054711 |
| small_int64_1d [read_full] | read_full | 0.000537 | fitsio:fitsio_torch | 0.000277 | 193.2 | 1.938 | 93.78 | off | NRC-054711 |
| small_float32_1d [read_full @ mps] | read_full | 0.001721 | fitsio:fitsio_torch_device_specialized | 0.000910 | 244.9 | 1.891 | 89.10 | off | NRC-054711 |
| timeseries_frame_003 [read_full] | read_full | 0.001308 | fitsio:fitsio_torch | 0.000704 | 185.1 | 1.857 | 85.75 | off | NRC-054711 |
| tiny_int64_1d [read_full] | read_full | 0.000421 | fitsio:fitsio_torch | 0.000232 | 181.0 | 1.814 | 81.41 | off | NRC-054711 |
| scaled_medium [read_full @ mps] | read_full | 0.005024 | fitsio:fitsio_torch_device_specialized | 0.002894 | 304.4 | 1.736 | 73.63 | off | NRC-054711 |
| scaled_large [read_full @ mps] | read_full | 0.012783 | fitsio:fitsio_torch_device_specialized | 0.008483 | 247.3 | 1.507 | 50.69 | off | NRC-054711 |
| multi_mef_10ext [cutout_100x100] | cutout_100x100 | 0.001082 | fitsio:fitsio_torch | 0.000735 | 181.7 | 1.471 | 47.11 | n/a | NRC-054711 |
| small_int16_2d [read_full @ mps] | read_full | 0.000638 | fitsio:fitsio_torch_device_specialized | 0.000440 | 213.0 | 1.449 | 44.88 | off | NRC-054711 |
| tiny_float32_2d [read_full @ mps] | read_full | 0.000788 | fitsio:fitsio_torch_device_specialized | 0.000576 | 143.2 | 1.369 | 36.87 | off | NRC-054711 |
| small_int64_1d [read_full @ mps] | read_full | 0.000660 | fitsio:fitsio_torch_device_specialized | 0.000488 | 188.7 | 1.352 | 35.25 | off | NRC-054711 |
| timeseries_frame_002 [read_full @ mps] | read_full | 0.001086 | fitsio:fitsio_torch_device_specialized | 0.000857 | 142.9 | 1.267 | 26.70 | off | NRC-054711 |
| tiny_int64_3d [read_full @ mps] | read_full | 0.000762 | fitsio:fitsio_torch_device_specialized | 0.000613 | 140.2 | 1.242 | 24.23 | off | NRC-054711 |
| small_int8_1d [read_full @ mps] | read_full | 0.000658 | fitsio:fitsio_torch_device_specialized | 0.000536 | 198.3 | 1.228 | 22.81 | off | NRC-054711 |
| small_int32_1d [read_full @ mps] | read_full | 0.001111 | fitsio:fitsio_torch_device_specialized | 0.000917 | 188.2 | 1.211 | 21.11 | off | NRC-054711 |
| medium_int32_2d [read_full @ mps] | read_full | 0.003325 | fitsio:fitsio_torch_device_specialized | 0.002777 | 232.7 | 1.198 | 19.76 | off | NRC-054711 |
| large_float32_1d [read_full @ mps] | read_full | 0.005873 | astropy:astropy_torch_device_specialized | 0.004914 | 217.7 | 1.195 | 19.51 | on | NRC-054711 |
| compressed_rice_1 [cutout_100x100] | cutout_100x100 | 0.003917 | fitsio:fitsio_torch | 0.003312 | 182.0 | 1.183 | 18.28 | n/a | NRC-054711 |
| timeseries_frame_000 [read_full @ mps] | read_full | 0.000890 | fitsio:fitsio_torch_device_specialized | 0.000765 | 142.4 | 1.163 | 16.31 | off | NRC-054711 |
| tiny_int8_3d [read_full] | read_full | 0.000287 | fitsio:fitsio_torch | 0.000249 | 180.8 | 1.153 | 15.28 | off | NRC-054711 |
| medium_uint16_2d [read_full @ mps] | read_full | 0.002660 | fitsio:fitsio_torch_device_specialized | 0.002356 | 185.5 | 1.129 | 12.88 | off | NRC-054711 |
| compressed_rice_1 [read_full @ mps] | read_full | 0.021945 | fitsio:fitsio_torch_device_specialized | 0.019743 | 188.1 | 1.112 | 11.15 | off | NRC-054711 |
| small_int16_3d [read_full @ mps] | read_full | 0.001798 | astropy:astropy_torch_device_specialized | 0.001619 | 215.0 | 1.110 | 11.03 | on | NRC-054711 |
| large_float32_2d [read_full @ mps] | read_full | 0.008269 | fitsio:fitsio_torch_device_specialized | 0.007631 | 267.2 | 1.084 | 8.36 | off | NRC-054711 |
| tiny_float32_1d [read_full @ mps] | read_full | 0.000644 | fitsio:fitsio_torch_device_specialized | 0.000598 | 143.4 | 1.078 | 7.80 | off | NRC-054711 |
| large_float32_1d [read_full @ mps] | read_full | 0.004032 | fitsio:fitsio_torch_device_specialized | 0.003749 | 211.4 | 1.076 | 7.56 | off | NRC-054711 |
| medium_float32_3d [read_full @ mps] | read_full | 0.004767 | fitsio:fitsio_torch_device_specialized | 0.004456 | 303.7 | 1.070 | 6.99 | off | NRC-054711 |
| small_int64_2d [read_full @ mps] | read_full | 0.000712 | fitsio:fitsio_torch_device_specialized | 0.000684 | 188.7 | 1.041 | 4.14 | off | NRC-054711 |
| small_int64_2d [read_full @ mps] | read_full | 0.002733 | astropy:astropy_torch_device_specialized | 0.002636 | 211.5 | 1.037 | 3.66 | on | NRC-054711 |
| tiny_int16_1d [read_full @ mps] | read_full | 0.000722 | fitsio:fitsio_torch_device_specialized | 0.000701 | 143.0 | 1.030 | 3.01 | off | NRC-054711 |
| medium_int64_3d [read_full @ mps] | read_full | 0.007490 | fitsio:fitsio_torch_device_specialized | 0.007314 | 299.2 | 1.024 | 2.41 | off | NRC-054711 |
| medium_uint32_2d [read_full @ mps] | read_full | 0.003970 | fitsio:fitsio_torch_device_specialized | 0.003882 | 216.0 | 1.023 | 2.27 | off | NRC-054711 |
| compressed_rice_1 [read_full @ mps] | read_full | 0.018417 | fitsio:fitsio_torch_device_specialized | 0.018023 | 214.0 | 1.022 | 2.19 | on | NRC-054711 |
| small_int32_3d [read_full @ mps] | read_full | 0.000758 | fitsio:fitsio_torch_device_specialized | 0.000747 | 188.6 | 1.015 | 1.53 | off | NRC-054711 |
| medium_int8_1d [read_full] | read_full | 0.000408 | fitsio:fitsio_torch | 0.000402 | 338.7 | 1.014 | 1.44 | off | NRC-054711 |
| compressed_rice_1 [read_full] | read_full | 0.016208 | fitsio:fitsio_torch | 0.016109 | 218.7 | 1.006 | 0.62 | on | NRC-054711 |
| compressed_gzip_2 [read_full @ mps] | read_full | 0.043027 | fitsio:fitsio_torch_device_specialized | 0.042796 | 217.8 | 1.005 | 0.54 | off | NRC-054711 |

### FITSTABLE - smart

| Case | Operation | TorchFits (s) | Winner | Winner (s) | Process peak RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| varlen_100000 [predicate_filter_selective] | predicate_filter_selective | 0.007270 | fitsio:fitsio_torch | 0.004492 | 206.6 | 1.618 | 61.85 | off | NRC-054711 |
| typed_100000 [predicate_filter] | predicate_filter | 0.008350 | fitsio:fitsio_torch | 0.005370 | 188.9 | 1.555 | 55.50 | off | NRC-054711 |
| narrow_10000 [read_full] | read_full | 0.000856 | fitsio:fitsio_torch | 0.000600 | 229.0 | 1.427 | 42.70 | off | NRC-054711 |
| wide_100000 [predicate_filter] | predicate_filter | 0.023139 | fitsio:fitsio_torch | 0.020506 | 307.8 | 1.128 | 12.84 | off | NRC-054711 |
| wide_100000 [predicate_filter_selective] | predicate_filter_selective | 0.022765 | fitsio:fitsio_torch | 0.022244 | 274.7 | 1.023 | 2.34 | off | NRC-054711 |
| narrow_1000000 [read_full] | read_full | 0.020491 | fitsio:fitsio_torch | 0.020116 | 362.8 | 1.019 | 1.86 | off | NRC-054711 |
| wide_10000 [predicate_filter_selective] | predicate_filter_selective | 0.002732 | fitsio:fitsio_torch | 0.002723 | 248.4 | 1.003 | 0.32 | off | NRC-054711 |

### FITSTABLE - specialized

| Case | Operation | TorchFits (s) | Winner | Winner (s) | Process peak RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| narrow_100000 [predicate_filter_selective] | predicate_filter_selective | 0.016426 | fitsio:fitsio | 0.007303 | 241.7 | 2.249 | 124.93 | off | NRC-054711 |
| narrow_1000000 [predicate_filter_selective] | predicate_filter_selective | 0.036221 | astropy:astropy | 0.022344 | 331.7 | 1.621 | 62.10 | off | NRC-054711 |
| mixed_1000000 [predicate_filter_selective] | predicate_filter_selective | 0.048531 | astropy:astropy | 0.032246 | 522.4 | 1.505 | 50.50 | off | NRC-054711 |
| typed_100000 [predicate_filter_selective] | predicate_filter_selective | 0.009028 | fitsio:fitsio | 0.006308 | 187.4 | 1.431 | 43.11 | off | NRC-054711 |
| typed_100000 [predicate_filter] | predicate_filter | 0.009320 | astropy:astropy | 0.006627 | 185.1 | 1.406 | 40.63 | off | NRC-054711 |
| varlen_10000 [predicate_filter_selective] | predicate_filter_selective | 0.002217 | fitsio:fitsio | 0.001671 | 223.7 | 1.327 | 32.65 | off | NRC-054711 |
| mixed_1000000 [predicate_filter] | predicate_filter | 0.049783 | astropy:astropy | 0.042856 | 533.7 | 1.162 | 16.16 | off | NRC-054711 |
| narrow_1000000 [predicate_filter] | predicate_filter | 0.037313 | astropy:astropy | 0.032443 | 329.6 | 1.150 | 15.01 | off | NRC-054711 |
| wide_100000 [predicate_filter_selective] | predicate_filter_selective | 0.022416 | fitsio:fitsio | 0.019498 | 274.8 | 1.150 | 14.96 | off | NRC-054711 |
| varlen_100000 [predicate_filter_selective] | predicate_filter_selective | 0.004710 | fitsio:fitsio | 0.004338 | 203.9 | 1.086 | 8.58 | off | NRC-054711 |
| ascii_10000 [predicate_filter_selective] | predicate_filter_selective | 0.001381 | fitsio:fitsio | 0.001313 | 167.8 | 1.051 | 5.13 | off | NRC-054711 |
| ascii_10000 [predicate_filter] | predicate_filter | 0.001380 | fitsio:fitsio | 0.001328 | 169.8 | 1.039 | 3.91 | off | NRC-054711 |

## Notes

- Strict mmap fairness is enforced in comparable sets. Rows with unmatched mmap controls are marked `SKIPPED`.
- `Process peak RSS` is the interpreter process's peak resident set, sampled by `bench_timing._RssPeakSampler`. It is the same for every method in a comparison group and is **not** a per-library memory figure; ranking is time-based.
- Rankings are family-specific and never mix smart vs specialized method families.
