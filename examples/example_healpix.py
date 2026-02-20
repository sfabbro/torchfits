#!/ reentry-compatible/env python3
"""Example of basic HEALPix operations using torchfits.sphere."""

import torch
import torchfits.sphere as sphere


def main():
    # 1. Setup NSIDE
    nside = 128
    npix = 12 * nside**2
    print(f"HEALPix NSIDE={nside}, NPIX={npix}")

    # 2. Convert coordinates to pixel indices
    # Using (theta, phi) in radians
    theta = torch.tensor([0.1, 0.5, 1.0, 1.5], dtype=torch.float64)
    phi = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)

    pix = sphere.ang2pix(nside, theta, phi, lonlat=False)
    print("\nCoordinates (theta, phi) to Pixels (RING):")
    for t, p, pi in zip(theta, phi, pix):
        print(f"  theta={t:.2f}, phi={p:.2f} -> pixel={pi.item()}")

    # 3. Convert pixels back to coordinates
    theta_out, phi_out = sphere.pix2ang(nside, pix, lonlat=False)
    print("\nPixels back to Coordinates:")
    for pi, t, p in zip(pix, theta_out, phi_out):
        print(f"  pixel={pi.item()} -> theta={t:.2f}, phi={p:.2f}")

    # 4. Spherical Cap (Disc) Query
    # Find all pixels within a 15 degree radius of a center vector
    center_vec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)  # North pole
    radius_deg = 15.0
    radius_rad = torch.deg2rad(torch.tensor(radius_deg))

    # query_disc expects unit vector
    disc_pix = sphere.query_disc(nside, center_vec, radius_rad.item())
    print(f"\nQuery Disc (North Pole, radius={radius_deg} deg):")
    print(f"  Found {len(disc_pix)} pixels in disc.")

    # 5. Sample values from a map
    # Create a dummy map with a gradient
    m = torch.linspace(0, 100, npix, dtype=torch.float64)

    # Sample at specific locations
    sample_lon = torch.tensor([10.0, 20.0])
    sample_lat = torch.tensor([45.0, 60.0])
    values = sphere.sample_healpix_map(
        m, nside=nside, lon_deg=sample_lon, lat_deg=sample_lat
    )
    print("\nSampled Map Values:")
    for lon, la, v in zip(sample_lon, sample_lat, values):
        print(f"  lon={lon:.2f}, lat={la:.2f} -> value={v:.2f}")


if __name__ == "__main__":
    main()
