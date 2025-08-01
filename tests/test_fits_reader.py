import unittest
import torch
import torchfits
import numpy as np
import os
from astropy.io import fits

class TestFitsReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = "test_data"
        os.makedirs(cls.test_dir, exist_ok=True)

        # --- 2D Image ---
        cls.image_file = os.path.join(cls.test_dir, "test_image.fits")
        cls.image_data = np.arange(100, dtype=np.float32).reshape(10, 10)
        hdu = fits.PrimaryHDU(cls.image_data)
        hdu.header['TESTKEY'] = 'TESTVAL'
        hdu.writeto(cls.image_file, overwrite=True)

        # --- 3D Cube ---
        cls.cube_file = os.path.join(cls.test_dir, "test_cube.fits")
        cls.cube_data = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        fits.writeto(cls.cube_file, cls.cube_data, overwrite=True)

        # --- MEF File ---
        cls.mef_file = os.path.join(cls.test_dir, "test_mef.fits")
        primary_hdu = fits.PrimaryHDU()
        ext1 = fits.ImageHDU(np.arange(100, dtype=np.float32).reshape(10, 10), name='EXT1')
        hdul = fits.HDUList([primary_hdu, ext1])
        hdul.writeto(cls.mef_file, overwrite=True)

        # --- Table File ---
        cls.table_file = os.path.join(cls.test_dir, "test_table.fits")
        col1 = fits.Column(name='TARGET', format='10A', array=np.array(['NGC1001', 'NGC1002', 'NGC1003']))
        col2 = fits.Column(name='RA', format='D', array=np.array([120.1, 120.2, 120.3]), unit='deg')  # Double precision
        col3 = fits.Column(name='DEC', format='D', array=np.array([-30.1, -30.2, -30.3]), unit='deg') # Double precision
        cols = fits.ColDefs([col1, col2, col3])
        table_hdu = fits.BinTableHDU.from_columns(cols)
        table_hdu.writeto(cls.table_file, overwrite=True)


    @classmethod
    def tearDownClass(cls):
        for file in os.listdir(cls.test_dir):
            os.remove(os.path.join(cls.test_dir, file))
        os.rmdir(cls.test_dir)

    def test_context_manager(self):
        with torchfits.FITS(self.image_file) as f:
            self.assertIsNotNone(f)

    def test_read_full_image(self):
        data, header = torchfits.read(self.image_file)
        self.assertTrue(isinstance(data, torch.Tensor))
        self.assertEqual(data.dtype, torch.float32)
        self.assertEqual(data.shape, (10, 10))
        self.assertTrue(np.allclose(data.numpy(), self.image_data))
        self.assertEqual(header['TESTKEY'], 'TESTVAL')

    def test_read_image_cutout(self):
        data, _ = torchfits.read(self.image_file, start=[2, 3], shape=[4, 5])
        self.assertEqual(data.shape, (4, 5))
        self.assertTrue(np.allclose(data.numpy(), self.image_data[2:6, 3:8]))

    def test_read_cube_cutout(self):
        data, _ = torchfits.read(self.cube_file, start=[0, 1, 1], shape=[2, 2, 2])
        self.assertEqual(data.shape, (2, 2, 2))
        self.assertTrue(np.allclose(data.numpy(), self.cube_data[0:2, 1:3, 1:3]))

    def test_read_mef(self):
        data, _ = torchfits.read(self.mef_file, hdu='EXT1')
        self.assertEqual(data.shape, (10, 10))

    def test_read_full_table(self):
        data, _ = torchfits.read(self.table_file, hdu=2)
        self.assertIn('RA', data)
        self.assertIn('DEC', data)
        self.assertTrue(np.allclose(data['RA'].numpy(), [120.1, 120.2, 120.3]))
        self.assertTrue(np.allclose(data['DEC'].numpy(), [-30.1, -30.2, -30.3]))

    def test_read_table_columns(self):
        data, _ = torchfits.read(self.table_file, hdu=2, columns=['RA', 'DEC'])
        self.assertIn('RA', data)
        self.assertIn('DEC', data)
        self.assertNotIn('TARGET', data)

    def test_read_table_rows(self):
        data, _ = torchfits.read(self.table_file, hdu=2, start_row=1, num_rows=2)
        self.assertEqual(len(data['RA']), 2)
        self.assertTrue(np.allclose(data['RA'].numpy(), [120.2, 120.3]))

    def test_error_handling(self):
        with self.assertRaises(RuntimeError):
            torchfits.read("nonexistent.fits")

        with self.assertRaises(RuntimeError):
            torchfits.read(self.image_file, hdu=99)

if __name__ == '__main__':
    unittest.main()