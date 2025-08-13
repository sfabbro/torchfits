"""Micro-benchmark: random multi-file cutout batch with optional async GPU transfer.

Measures throughput of read_batch across many small cutouts from random files.

Usage (CPU baseline):
    python benchmarks/cutout_batch_micro.py

With async GPU (if CUDA available):
    TORCHFITS_ASYNC_GPU=1 python benchmarks/cutout_batch_micro.py --cuda

Environment knobs:
    TORCHFITS_BATCH_THREADS : limit parallel read threads (default=min(N,32))
    TORCHFITS_ASYNC_GPU     : if set to 1, force async GPU transfers even if spec.async_gpu False
"""
from __future__ import annotations
import argparse, os, time, statistics, tempfile
import torch, sys
import torchfits
from torchfits.dataset import BatchReadSpec, read_batch, generate_random_cutout_specs
import os
sys.path.append(os.path.dirname(__file__))
from bench_utils import format_table  # type: ignore


def make_files(n_files=6, shape=(512,512)):
    tmp = tempfile.TemporaryDirectory()
    paths = []
    for i in range(n_files):
        t = torch.randn(*shape)
        path = os.path.join(tmp.name, f"img_{i}.fits")
        torchfits.write(path, t, overwrite=True)
        paths.append(path)
    return tmp, paths


def run(rep=5, batch_size=32, cutout_shape=(64,64), cuda=False, async_flag=False):
    tmp, paths = make_files()
    device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
    specs = []
    for _ in range(batch_size):
        specs.extend(generate_random_cutout_specs(paths, hdu=0, shape=cutout_shape, n=1, device=device))
    # Mark async on each spec if requested
    if async_flag and device.startswith('cuda'):
        specs = [BatchReadSpec(**{**s.__dict__, 'async_gpu': True, 'device': device}) for s in specs]

    times = []
    for _ in range(rep):
        t0 = time.time()
        read_batch(specs, parallel=True, return_dict=False)
        torch.cuda.synchronize() if device.startswith('cuda') and torch.cuda.is_available() else None
        times.append((time.time() - t0) * 1000.0)
    return {
        'device': device,
        'async': async_flag,
        'ms_mean': statistics.mean(times),
        'ms_stdev': statistics.pstdev(times),
        'reps': rep,
        'batch_size': batch_size,
        'cutout_shape': cutout_shape,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cuda', action='store_true')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--rep', type=int, default=5)
    ap.add_argument('--size', type=int, default=64, help='square cutout size')
    args = ap.parse_args()

    res_cpu = run(rep=args.rep, batch_size=args.batch, cutout_shape=(args.size,args.size), cuda=False, async_flag=False)
    rows = [["torchfits", "CPU", f"{res_cpu['ms_mean']:.2f}", f"{res_cpu['ms_stdev']:.2f}", f"batch={args.batch}, sz={args.size}"]]
    if args.cuda and torch.cuda.is_available():
        res_gpu_sync = run(rep=args.rep, batch_size=args.batch, cutout_shape=(args.size,args.size), cuda=True, async_flag=False)
        rows.append(["torchfits", "GPU sync", f"{res_gpu_sync['ms_mean']:.2f}", f"{res_gpu_sync['ms_stdev']:.2f}", f"batch={args.batch}, sz={args.size}"])
        res_gpu_async = run(rep=args.rep, batch_size=args.batch, cutout_shape=(args.size,args.size), cuda=True, async_flag=True)
        rows.append(["torchfits", "GPU async", f"{res_gpu_async['ms_mean']:.2f}", f"{res_gpu_async['ms_stdev']:.2f}", f"batch={args.batch}, sz={args.size}"])
        extra = []
        if res_gpu_async['ms_mean'] < res_gpu_sync['ms_mean']:
            extra.append(f"async speedup {res_gpu_sync['ms_mean']/res_gpu_async['ms_mean']:.2f}x vs sync")
        print("\n== Batched cutouts ==")
        print(format_table(rows, headers=["Impl", "Mode", "mean ms", "stdev", "notes"]))
        if extra:
            print("(" + ", ".join(extra) + ")")
    else:
        print("\n== Batched cutouts ==")
        print(format_table(rows, headers=["Impl", "Mode", "mean ms", "stdev", "notes"]))

if __name__ == '__main__':
    main()
