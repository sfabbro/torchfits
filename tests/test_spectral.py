"""
Tests for spectral and data cube functionality.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

# Add src to path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from torchfits.spectral import (
    SpectralAxis, Spectrum1D, DataCube, SpectralReader
)


class TestSpectralAxis:
    """Test SpectralAxis functionality."""
    
    def test_wavelength_axis(self):
        """Test wavelength spectral axis."""
        wavelengths = torch.linspace(4000, 7000, 1000)  # Angstroms
        axis = SpectralAxis(wavelengths, 'Angstrom', 'WAVE')
        
        assert torch.allclose(axis.to_wavelength(), wavelengths)
        
        # Test frequency conversion
        freq = axis.to_frequency()
        assert freq.shape == wavelengths.shape
        assert torch.all(freq > 0)
    
    def test_frequency_axis(self):
        """Test frequency spectral axis."""
        frequencies = torch.linspace(1e14, 2e14, 1000)  # Hz
        axis = SpectralAxis(frequencies, 'Hz', 'FREQ')
        
        assert torch.allclose(axis.to_frequency(), frequencies)
        
        # Test wavelength conversion
        wave = axis.to_wavelength()
        assert wave.shape == frequencies.shape
        assert torch.all(wave > 0)
    
    def test_unit_conversions(self):
        """Test unit conversions."""
        # Test nm to Angstrom
        wave_nm = torch.tensor([400.0, 500.0, 600.0])  # nm
        axis = SpectralAxis(wave_nm, 'nm', 'WAVE')
        wave_angstrom = axis.to_wavelength()
        expected = wave_nm * 10.0
        assert torch.allclose(wave_angstrom, expected)


class TestSpectrum1D:
    """Test 1D spectrum functionality."""
    
    def create_test_spectrum(self, n_wave=1000, with_ivar=True, with_mask=True):
        """Create test spectrum."""
        wavelengths = torch.linspace(4000, 7000, n_wave)
        flux = torch.sin(wavelengths / 100) + torch.randn(n_wave) * 0.1
        
        spectral_axis = SpectralAxis(wavelengths, 'Angstrom', 'WAVE')
        
        ivar = None
        if with_ivar:
            ivar = torch.ones_like(flux) * 100  # High S/N
        
        mask = None
        if with_mask:
            mask = torch.ones_like(flux, dtype=torch.bool)
            # Mask out some bad pixels
            mask[100:110] = False
            mask[500:505] = False
        
        return Spectrum1D(flux, spectral_axis, ivar, mask)
    
    def test_spectrum_creation(self):
        """Test spectrum creation and validation."""
        spectrum = self.create_test_spectrum()
        
        assert spectrum.flux.ndim == 1
        assert len(spectrum.spectral_axis.values) == len(spectrum.flux)
        assert spectrum.ivar.shape == spectrum.flux.shape
        assert spectrum.mask.shape == spectrum.flux.shape
    
    def test_wavelength_frequency_access(self):
        """Test wavelength and frequency access."""
        spectrum = self.create_test_spectrum()
        
        wavelength = spectrum.wavelength
        frequency = spectrum.frequency
        
        assert wavelength.shape == spectrum.flux.shape
        assert frequency.shape == spectrum.flux.shape
        assert torch.all(wavelength > 0)
        assert torch.all(frequency > 0)
    
    def test_error_calculation(self):
        """Test error calculation from inverse variance."""
        spectrum = self.create_test_spectrum()
        
        error = spectrum.error
        assert error is not None
        assert error.shape == spectrum.flux.shape
        
        # Check that error = 1/sqrt(ivar)
        expected_error = torch.sqrt(1.0 / spectrum.ivar)
        assert torch.allclose(error, expected_error)
    
    def test_mask_application(self):
        """Test mask application."""
        spectrum = self.create_test_spectrum()
        original_length = len(spectrum.flux)
        
        masked_spectrum = spectrum.apply_mask()
        
        # Should have fewer pixels after masking
        assert len(masked_spectrum.flux) < original_length
        assert len(masked_spectrum.spectral_axis.values) == len(masked_spectrum.flux)
        
        # All remaining pixels should be good
        assert torch.all(masked_spectrum.mask)
    
    def test_resampling(self):
        """Test spectrum resampling."""
        spectrum = self.create_test_spectrum()
        
        # Create new wavelength grid
        new_wavelength = torch.linspace(4500, 6500, 500)
        resampled = spectrum.resample(new_wavelength)
        
        assert len(resampled.flux) == len(new_wavelength)
        assert torch.allclose(resampled.wavelength, new_wavelength)
        
        # Check that flux values are reasonable
        assert torch.all(torch.isfinite(resampled.flux))


class TestDataCube:
    """Test 3D data cube functionality."""
    
    def create_test_cube(self, shape=(100, 64, 64), spectral_axis_index=0):
        """Create test data cube."""
        data = torch.randn(shape)
        
        # Create spectral axis
        n_spectral = shape[spectral_axis_index]
        wavelengths = torch.linspace(4000, 7000, n_spectral)
        spectral_axis = SpectralAxis(wavelengths, 'Angstrom', 'WAVE')
        
        return DataCube(data, spectral_axis, spectral_axis_index=spectral_axis_index)
    
    def test_cube_creation(self):
        """Test cube creation and validation."""
        cube = self.create_test_cube()
        
        assert cube.data.ndim == 3
        assert cube.n_spectral == cube.data.shape[cube.spectral_axis_index]
        assert len(cube.spectral_axis.values) == cube.n_spectral
    
    def test_different_spectral_axes(self):
        """Test cubes with spectral axis in different positions."""
        for axis_idx in [0, 1, 2]:
            shape = [100, 64, 64]
            cube = self.create_test_cube(tuple(shape), axis_idx)
            
            assert cube.spectral_axis_index == axis_idx
            assert cube.n_spectral == shape[axis_idx]
    
    def test_spectrum_extraction(self):
        """Test extracting 1D spectra from cube."""
        cube = self.create_test_cube()
        
        # Extract spectrum at center
        y, x = 32, 32
        spectrum = cube.extract_spectrum(y, x)
        
        assert isinstance(spectrum, Spectrum1D)
        assert len(spectrum.flux) == cube.n_spectral
        assert torch.allclose(spectrum.spectral_axis.values, cube.spectral_axis.values)
    
    def test_spectral_collapse(self):
        """Test collapsing cube along spectral axis."""
        cube = self.create_test_cube()
        
        # Test different collapse methods
        for method in ['mean', 'sum', 'median', 'max']:
            collapsed = cube.collapse_spectral(method)
            
            expected_shape = list(cube.shape)
            expected_shape.pop(cube.spectral_axis_index)
            assert collapsed.shape == tuple(expected_shape)
    
    def test_wavelength_slice(self):
        """Test extracting 2D slice at specific wavelength."""
        cube = self.create_test_cube()
        
        # Extract slice at middle wavelength
        mid_wavelength = cube.spectral_axis.values[cube.n_spectral // 2]
        slice_2d = cube.extract_slice(mid_wavelength.item())
        
        expected_shape = list(cube.shape)
        expected_shape.pop(cube.spectral_axis_index)
        assert slice_2d.shape == tuple(expected_shape)
    
    def test_device_transfer(self):
        """Test moving cube to different device."""
        cube = self.create_test_cube()
        
        # Test CPU (should be no-op)
        cpu_cube = cube.to(torch.device('cpu'))
        assert cpu_cube.data.device == torch.device('cpu')
        
        # Test CUDA if available
        if torch.cuda.is_available():
            cuda_cube = cube.to(torch.device('cuda'))
            assert cuda_cube.data.device.type == 'cuda'


class TestSpectralReader:
    """Test reading spectral data from FITS files."""
    
    def create_test_spectrum_file(self):
        """Create test spectrum FITS file."""
        try:
            from astropy.io import fits
            from astropy.table import Table
        except ImportError:
            pytest.skip("astropy required for test file creation")
        
        # Create test spectrum data
        n_wave = 1000
        wavelength = np.linspace(4000, 7000, n_wave)
        flux = np.sin(wavelength / 100) + np.random.normal(0, 0.1, n_wave)
        ivar = np.ones_like(flux) * 100
        mask = np.ones_like(flux, dtype=bool)
        mask[100:110] = False  # Bad pixels
        
        # Create table
        table = Table({
            'WAVELENGTH': wavelength,
            'FLUX': flux,
            'IVAR': ivar,
            'MASK': mask
        })
        
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            table.write(f.name, format='fits', overwrite=True)
            return f.name
    
    def create_test_cube_file(self):
        """Create test data cube FITS file."""
        try:
            from astropy.io import fits
        except ImportError:
            pytest.skip("astropy required for test file creation")
        
        # Create test cube data
        shape = (100, 64, 64)  # wavelength, y, x
        data = np.random.randn(*shape).astype(np.float32)
        
        # Create HDU with spectral WCS
        hdu = fits.PrimaryHDU(data)
        hdu.header['CTYPE3'] = 'WAVE'
        hdu.header['CRVAL3'] = 4000.0  # Angstroms
        hdu.header['CDELT3'] = 30.0    # Angstrom/pixel
        hdu.header['CRPIX3'] = 1.0
        hdu.header['CUNIT3'] = 'Angstrom'
        
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            hdu.writeto(f.name, overwrite=True)
            return f.name
    
    @pytest.mark.skip("Requires full torchfits implementation")
    def test_read_spectrum_1d(self):
        """Test reading 1D spectrum from FITS table."""
        filepath = self.create_test_spectrum_file()
        
        try:
            spectrum = SpectralReader.read_spectrum_1d(filepath, hdu=1)
            
            assert isinstance(spectrum, Spectrum1D)
            assert spectrum.flux.ndim == 1
            assert spectrum.ivar is not None
            assert spectrum.mask is not None
            
            # Check wavelength range
            wavelength = spectrum.wavelength
            assert wavelength[0] >= 4000
            assert wavelength[-1] <= 7000
            
        finally:
            os.unlink(filepath)
    
    @pytest.mark.skip("Requires full torchfits implementation")
    def test_read_data_cube(self):
        """Test reading 3D data cube from FITS image."""
        filepath = self.create_test_cube_file()
        
        try:
            cube = SpectralReader.read_data_cube(filepath)
            
            assert isinstance(cube, DataCube)
            assert cube.data.ndim == 3
            assert cube.spectral_axis_index == 0  # Should detect wavelength axis
            
            # Check spectral axis
            wavelength = cube.spectral_axis.to_wavelength()
            assert wavelength[0] >= 4000
            assert wavelength[-1] <= 7000
            
        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])