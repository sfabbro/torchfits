#!/usr/bin/env bash
# Runtime library search path for the bench-gpu env, without hardcoding the
# python version. torch's bundled libtorch_cuda/libc10_cuda (dlopen-time only
# for the CPU-built torchfits._C) and the nvidia-*-cu12 wheels live under
# site-packages/<python-version>/... — the loader glob covers any version.
# Source from bench-gpu tasks, e.g.:
#   bash -c 'source scripts/gpu-env-loader.sh && <command>'
set -u

SP_ROOT="${CONDA_PREFIX:-}/lib"
LD_EXTRA=""
for d in \
  "${SP_ROOT}"/python3.*/site-packages/torch/lib \
  "${SP_ROOT}"/python3.*/site-packages/nvidia/cuda_runtime/lib \
  "${SP_ROOT}"/python3.*/site-packages/nvidia/nccl/lib \
  "${SP_ROOT}"/python3.*/site-packages/nvidia/cublas/lib \
  "${SP_ROOT}"/python3.*/site-packages/nvidia/cusparse/lib \
  "${SP_ROOT}"/python3.*/site-packages/nvidia/cusolver/lib \
  "${SP_ROOT}"/python3.*/site-packages/nvidia/cudnn/lib \
  "${SP_ROOT}"/python3.*/site-packages/nvidia/cufft/lib; do
  if [[ -d "${d}" ]]; then
    LD_EXTRA="${LD_EXTRA:+${LD_EXTRA}:}${d}"
  fi
done
if [[ -n "${LD_EXTRA}" ]]; then
  export LD_LIBRARY_PATH="${LD_EXTRA}:${LD_LIBRARY_PATH:-}"
fi
