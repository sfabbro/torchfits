import torch
import torchfits
import numpy as np
import os
from astropy.io import fits

def create_test_files(data_dir):
    os.makedirs(data_dir, exist_ok=True)

    # --- 2D Image (RA/Dec) ---
    image_file = os.path.join(data_dir, "test_image_2d.fits")
    image_data = np.random.rand(100, 100).astype(np.float32)
    hdu = fits.PrimaryHDU(image_data)
    hdu.header['CTYPE1'] = 'RA---TAN'  # Standard RA/Dec with TAN projection
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CRVAL1'] = 200.0      # Reference value (RA)
    hdu.header['CRVAL2'] = 45.0       # Reference value (Dec)
    hdu.header['CRPIX1'] = 50.0       # Reference pixel (x)
    hdu.header['CRPIX2'] = 50.0       # Reference pixel (y)
    hdu.header['CDELT1'] = -0.01      # Degrees per pixel (RA)
    hdu.header['CDELT2'] = 0.01       # Degrees per pixel (Dec)
    hdu.writeto(image_file, overwrite=True)

    # --- 1D Spectrum (Wavelength) ---
    spectrum_file = os.path.join(data_dir, "test_spectrum_1d.fits")
    wavelengths = np.linspace(4000, 7000, 1000)  # Angstroms
    flux = np.random.rand(1000).astype(np.float32)
    hdu = fits.PrimaryHDU(flux)
    hdu.header['CTYPE1'] = 'WAVE'       # Wavelength
    hdu.header['CUNIT1'] = 'Angstrom'   # Units
    hdu.header['CRVAL1'] = 4000.0     # Starting wavelength
    hdu.header['CRPIX1'] = 1.0        # Reference pixel
    hdu.header['CDELT1'] = 3.0        # Wavelength per pixel
    hdu.writeto(spectrum_file, overwrite=True)

    # --- 3D Cube (RA/Dec/Wavelength) ---
    cube_file = os.path.join(data_dir, "test_cube_3d.fits")
    cube_data = np.random.rand(10, 20, 30).astype(np.float32)
    hdu = fits.PrimaryHDU(cube_data)
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CTYPE3'] = 'WAVE'
    hdu.header['CUNIT3'] = 'Angstrom'
    hdu.header['CRVAL1'] = 200.0
    hdu.header['CRVAL2'] = 45.0
    hdu.header['CRVAL3'] = 5000.0
    hdu.header['CRPIX1'] = 5.0
    hdu.header['CRPIX2'] = 10.0
    hdu.header['CRPIX3'] = 1.0
    hdu.header['CDELT1'] = -0.01
    hdu.header['CDELT2'] = 0.01
    hdu.header['CDELT3'] = 5.0
    hdu.writeto(cube_file, overwrite=True)

def main():
    data_dir = "data_wcs_examples"
    create_test_files(data_dir)

    # --- 2D Image Example ---
    print("\n--- 2D Image (RA/Dec) ---")
    image_file = os.path.join(data_dir, "test_image_2d.fits")
    data, header = torchfits.read(image_file)

    # World to Pixel
    world_coords = torch.tensor([[200.0, 45.0], [199.95, 45.05]], dtype=torch.float64)
    pixel_coords, status = torchfits.world_to_pixel(world_coords, header)
    print(f"World Coordinates: {world_coords}")
    print(f"Pixel Coordinates: {pixel_coords}")

    # Pixel to World
    pixel_coords = torch.tensor([[50.0, 50.0], [45.0, 55.0]], dtype=torch.float64)
    world_coords, status = torchfits.pixel_to_world(pixel_coords, header)
    print(f"Pixel Coordinates: {pixel_coords}")
    print(f"World Coordinates: {world_coords}")

    # --- 1D Spectrum Example ---
    print("\n--- 1D Spectrum (Wavelength) ---")
    spectrum_file = os.path.join(data_dir, "test_spectrum_1d.fits")
    data, header = torchfits.read(spectrum_file)

    # World to Pixel
    world_coords = torch.tensor([[4000.0], [5500.0]], dtype=torch.float64)  # Angstroms
    pixel_coords, status = torchfits.world_to_pixel(world_coords, header)
    print(f"Wavelengths: {world_coords}")
    print(f"Pixel Indices: {pixel_coords}")

    # Pixel to World
    pixel_coords = torch.tensor([[1.0], [501.0]], dtype=torch.float64)  # Pixel index
    world_coords, status = torchfits.pixel_to_world(pixel_coords, header)
    print(f"Pixel Indices: {pixel_coords}")
    print(f"Wavelengths: {world_coords}")


    # --- 3D Cube Example ---
    print("\n--- 3D Cube (RA/Dec/Wavelength) ---")
    cube_file = os.path.join(data_dir, "test_cube_3d.fits")
    data, header = torchfits.read(cube_file)

    # World to Pixel
    world_coords = torch.tensor([[200.0, 45.0, 5000.0], [199.95, 45.05, 5025.0]], dtype=torch.float64)
    pixel_coords, _ = torchfits.world_to_pixel(world_coords, header)
    print(f"World Coordinates: {world_coords}")
    print(f"Pixel Coordinates: {pixel_coords}")

    # Pixel to World
    pixel_coords = torch.tensor([[5.0, 10.0, 1.0], [4.0, 11.0, 6.0]], dtype=torch.float64)
    world_coords, _ = torchfits.pixel_to_world(pixel_coords, header)
    print(f"Pixel Coordinates: {pixel_coords}")
    print(f"World Coordinates: {world_coords}")

if __name__ == "__main__":
    main()
    