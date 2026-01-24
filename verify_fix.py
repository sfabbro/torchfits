
import torch
import numpy as np
from astropy.wcs import WCS as AstropyWCS
from astropy.io import fits
from torchfits.wcs import WCS as TorchWCS
from torchfits.hdu import Header

def verify():
    # 1. Setup Header with CD=Identity (Requires lonpole=0)
    h = fits.Header()
    h['SIMPLE'] = 'T'
    h['BITPIX'] = -32
    h['NAXIS'] = 2
    h['NAXIS1'] = 100
    h['NAXIS2'] = 100
    h['CTYPE1'] = 'RA---TAN-SIP'
    h['CTYPE2'] = 'DEC--TAN-SIP'
    h['CRVAL1'] = 0.0
    h['CRVAL2'] = 0.0
    h['CRPIX1'] = 50.0
    h['CRPIX2'] = 50.0
    h['CD1_1'] = 1.0
    h['CD1_2'] = 0.0
    h['CD2_1'] = 0.0
    h['CD2_2'] = 1.0
    h['A_ORDER'] = 2
    h['A_2_0'] = 0.1
    h['B_ORDER'] = 2

    # 2. Objects
    aw = AstropyWCS(h)
    h_dict = dict(h)
    tw = TorchWCS(Header(h_dict))

    print(f"Torch Lonpole: {tw._lonpole}")
    print(f"Torch Latpole: {tw._latpole}")

    print(f"Astro Lonpole: {aw.wcs.lonpole}")
    print(f"Astro Latpole: {aw.wcs.latpole}")
    print(f"Astro CD: {aw.wcs.cd}")
    print(f"Astro PC: {aw.wcs.pc}")
    print(f"Astro CDELT: {aw.wcs.cdelt}")

    # 3. Test Point
    pixel = torch.tensor([[60.0, 50.0]], dtype=torch.float64)

    world_t = tw.pixel_to_world(pixel)
    world_a = aw.pixel_to_world_values(pixel[0,0].item(), pixel[0,1].item())

    print(f"Torch: {world_t[0].tolist()}")
    print(f"Astro: {world_a}")

    diff_ra = abs(world_t[0,0].item() - world_a[0])
    diff_dec = abs(world_t[0,1].item() - world_a[1])

    # Fix circular diff
    if diff_ra > 180: diff_ra = 360 - diff_ra

    dist = np.sqrt(diff_ra**2 + diff_dec**2) * 3600.0
    print(f"Diff: {dist:.6f} arcsec")

    if dist < 1.0:
        print("SUCCESS: Error is negligible.")
    else:
        print("FAILURE: Error is still large.")

if __name__ == "__main__":
    verify()
