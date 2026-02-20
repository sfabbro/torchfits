#!/usr/bin/env python3
"""Example of Spherical Harmonic Transforms using torchfits.sphere.

This example demonstrates:
1. Generating a random Gaussian map from a power spectrum.
2. Analyzing the map to spherical harmonic coefficients (alms).
3. Smoothing the map in harmonic space.
4. Synthesizing the smoothed map back to pixel space.
"""

import torch
import torchfits.sphere as sphere


def main():
    nside = 32
    lmax = 3 * nside - 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SHT Example with NSIDE={nside}, LMAX={lmax} on {device}")

    # 1. Generate a power spectrum (Cls)
    # Just a simple decaying power spectrum for demonstration
    ell = torch.arange(lmax + 1, dtype=torch.float64, device=device)
    cls = 1.0 / (ell + 1) ** 2

    # 2. Generate a random Gaussian map from the Cls
    # synfast returns a map in RING ordering by default
    print("\nGenerating random map from power spectrum...")
    m = sphere.synfast(cls, nside=nside, lmax=lmax)
    print(f"  Generated map with shape {m.shape}")
    print(f"  Map finite: {torch.isfinite(m).all().item()}")
    print(f"  Map mean: {m.mean().item():.4f}, std: {m.std().item():.4f}")

    # 3. Analyze the map: Map -> alms
    # This uses the optimized C++ recurrence kernels
    print("\nAnalyzing map to spherical harmonic coefficients (map2alm)...")
    alms = sphere.map2alm(m, lmax=lmax)
    print(f"  Computed alms with shape {alms.shape}")
    print(f"  Alms finite: {torch.isfinite(alms).all().item()}")

    # 4. Filter in harmonic space (Smoothing)
    # Apply a Gaussian beam to smooth the map
    fwhm_deg = 5.0
    fwhm_rad = torch.deg2rad(torch.tensor(fwhm_deg)).item()
    print(f"\nSmoothing map with FWHM={fwhm_deg} degrees...")

    # beam is the b_l coefficients
    bl = sphere.gaussian_beam(fwhm_rad, lmax=lmax)

    # Multiply alms by beam coefficients
    alms_smoothed = sphere.almxfl(alms, bl)
    print(f"  Smoothed alms finite: {torch.isfinite(alms_smoothed).all().item()}")

    # 5. Synthesize the smoothed map: alms -> Map
    print("\nSynthesizing smoothed map (alm2map)...")
    m_smoothed = sphere.alm2map(alms_smoothed, nside=nside, lmax=lmax)
    print(f"  Smoothed map finite: {torch.isfinite(m_smoothed).all().item()}")

    # 6. Verify smoothing
    print("\nSmoothing Results:")
    print(f"  Original map StdDev: {m.std().item():.4f}")
    print(f"  Smoothed map StdDev: {m_smoothed.std().item():.4f}")
    print("  (StdDev reduction expected due to smoothing)")


if __name__ == "__main__":
    main()
