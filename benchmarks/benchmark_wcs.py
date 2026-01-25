import time
import torch
import numpy as np
from astropy.wcs import WCS as AstropyWCS
from astropy.io import fits
from torchfits.wcs import WCS as TorchWCS
from torchfits.hdu import Header


def create_header(sip=False):
    h = fits.Header()
    h["SIMPLE"] = "T"
    h["BITPIX"] = -32
    h["NAXIS"] = 2
    h["NAXIS1"] = 2048
    h["NAXIS2"] = 2048
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"
    h["CRVAL1"] = 150.0
    h["CRVAL2"] = 20.0
    h["CRPIX1"] = 1024.5
    h["CRPIX2"] = 1024.5
    h["CD1_1"] = -0.00027
    h["CD1_2"] = 0.0
    h["CD2_1"] = 0.0
    h["CD2_2"] = 0.00027

    if sip:
        h["CTYPE1"] = "RA---TAN-SIP"
        h["CTYPE2"] = "DEC--TAN-SIP"
        h["A_ORDER"] = 2
        h["A_2_0"] = 1e-4
        h["B_ORDER"] = 2
        h["B_0_2"] = 1e-4
        h["AP_ORDER"] = 2  # Inverse coefficients for approximate roundtrip
        h["BP_ORDER"] = 2

    return h


def benchmark(n_coords=1_000_000, device="cpu", sip=False):
    print(f"\n--- Benchmarking WCS (SIP={sip}) on {device} with {n_coords} coords ---")

    # Setup Astropy WCS
    fits_header = create_header(sip)
    aw = AstropyWCS(fits_header)

    # Setup Torchfits WCS
    # Convert fits.Header to dict/Header wrapper
    h_dict = dict(fits_header)
    th = Header(h_dict)
    tw = TorchWCS(th)

    if device == "cuda" and torch.cuda.is_available():
        tw.to("cuda")

    # Generate random pixels (0-based)
    pixels_np = np.random.rand(n_coords, 2) * 2000.0
    pixels_t = torch.tensor(pixels_np, dtype=torch.float64)
    if device == "cuda" and torch.cuda.is_available():
        pixels_t = pixels_t.to("cuda")

    # --- Correctness Check ---
    # Run Astropy (CPU, 0-based)
    # Astropy: pixel_to_world_values
    t0 = time.time()
    ra_astro, dec_astro = aw.pixel_to_world_values(pixels_np[:, 0], pixels_np[:, 1])
    t_astro = time.time() - t0

    # --- DEBUG INFO ---
    print(f"\n[DEBUG] SIP={sip}")
    print(f"[DEBUG] Astropy CRVAL: {aw.wcs.crval}")
    print(f"[DEBUG] Torch   CRVAL: {tw.crval.cpu().numpy()}")
    print(f"[DEBUG] Astropy CRPIX: {aw.wcs.crpix}")
    print(f"[DEBUG] Torch   CRPIX: {tw.crpix.cpu().numpy()}")
    print(f"[DEBUG] Astropy CDELT: {aw.wcs.cdelt}")
    print(f"[DEBUG] Torch   CDELT: {tw.cdelt.cpu().numpy()}")
    try:
        print(f"[DEBUG] Astropy PC:\n{aw.wcs.pc}")
    except AttributeError:
        print(f"[DEBUG] Astropy PC: Not present (using CD?)")

    print(f"[DEBUG] Torch   PC:\n{tw.pc.cpu().numpy()}")
    print(f"[DEBUG] Astropy Lonpole: {aw.wcs.lonpole}")
    print(f"[DEBUG] Torch   Lonpole: {tw._lonpole}")

    print(f"[DEBUG] Pixel[0]: {pixels_np[0]}")
    print(f"[DEBUG] Astropy World[0]: {ra_astro[0]}, {dec_astro[0]}")

    # Run Torchfits
    with torch.no_grad():
        t0 = time.time()
        world_t = tw.pixel_to_world(pixels_t)
        if device == "cuda":
            torch.cuda.synchronize()
        t_torch = time.time() - t0

    world_np = world_t.cpu().numpy()
    print(f"[DEBUG] Torch   World[0]: {world_np[0, 0]}, {world_np[0, 1]}")

    # Diff
    ra_diff = np.abs(world_np[:, 0] - ra_astro)
    dec_diff = np.abs(world_np[:, 1] - dec_astro)

    # Handle RA wrap around 360
    ra_diff = np.minimum(ra_diff, 360 - ra_diff)

    max_diff = np.max(np.sqrt(ra_diff**2 + dec_diff**2)) * 3600.0  # arcsec
    print(f"Max difference vs Astropy: {max_diff:.2e} arcsec")

    # assert max_diff < 1.0, "Result mismatch > 1 arcsec!"

    # --- Performance ---
    print(f"Astropy time: {t_astro:.4f}s ({n_coords/t_astro:.2e} coords/s)")
    print(f"Torchfits time: {t_torch:.4f}s ({n_coords/t_torch:.2e} coords/s)")
    print(f"Speedup: {t_astro/t_torch:.2f}x")

    # --- Round Trip Check ---
    with torch.no_grad():
        pix_rec = tw.world_to_pixel(world_t)
        if device == "cuda":
            torch.cuda.synchronize()

    pix_diff = torch.norm(pix_rec - pixels_t, dim=1).max().item()
    print(f"\nRound-trip error (World->Pix): {pix_diff:.2e} pixels")


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"

    # CPU Benchmark
    benchmark(sip=False, device="cpu")
    benchmark(sip=True, device="cpu")

    # GPU Benchmark (if available)
    if dev != "cpu":
        benchmark(sip=False, device=dev)
        benchmark(sip=True, device=dev)
