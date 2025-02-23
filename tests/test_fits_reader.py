import unittest
import torch
import torchfits
import numpy as np
import os
import pytest  # Using pytest
from astropy.io import fits
from astropy.table import Table

class TestFitsReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create test FITS files (image, cube, binary table, MEF) *once*.
        cls.test_dir = "test_data"
        os.makedirs(cls.test_dir, exist_ok=True)

        # --- 2D Image ---
        cls.image_file = os.path.join(cls.test_dir, "test_image.fits")
        cls.image_data = np.arange(100, dtype=np.float32).reshape(10, 10)
        hdu = fits.PrimaryHDU(cls.image_data)
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'
        hdu.header['CRVAL1'] = 202.5
        hdu.header['CRVAL2'] = 47.5
        hdu.header['CRPIX1'] = 5.0
        hdu.header['CRPIX2'] = 5.0
        hdu.header['CDELT1'] = -0.001
        hdu.header['CDELT2'] = 0.001
        hdu.header['OBJECT'] = 'Test Image'
        hdu.writeto(cls.image_file, overwrite=True)

        # --- 3D Cube ---
        cls.cube_file = os.path.join(cls.test_dir, "test_cube.fits")
        cls.cube_data = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        hdu = fits.PrimaryHDU(cls.cube_data)
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'
        hdu.header['CTYPE3'] = 'VELO-LSR'
        hdu.header['CRVAL1'] = 202.5
        hdu.header['CRVAL2'] = 47.5
        hdu.header['CRVAL3'] = 1000.0
        hdu.header['CRPIX1'] = 2.0
        hdu.header['CRPIX2'] = 2.0
        hdu.header['CRPIX3'] = 1.0
        hdu.header['CDELT1'] = -0.001
        hdu.header['CDELT2'] = 0.001
        hdu.header['CDELT3'] = 2.5
        hdu.writeto(cls.cube_file, overwrite=True)

        # --- 1D Image ---
        cls.image_1d_file = os.path.join(cls.test_dir, "test_image_1d.fits")
        cls.image_1d_data = np.arange(10, dtype=np.float32)
        hdu = fits.PrimaryHDU(cls.image_1d_data)
        hdu.writeto(cls.image_1d_file, overwrite=True)

        # --- Binary Table ---
        cls.table_file = os.path.join(cls.test_dir, "test_table.fits")
        names = ['ra', 'dec', 'flux', 'id', 'flag', 'comments']
        formats = ['D', 'D', 'E', 'J', 'B', '20A']
        data = {
            'ra': np.array([200.0, 201.0, 202.0], dtype=np.float64),
            'dec': np.array([45.0, 46.0, 47.0], dtype=np.float64),
            'flux': np.array([1.0, 2.0, 3.0], dtype=np.float32),
            'id': np.array([1, 2, 3], dtype=np.int32),
            'flag': np.array([1, 0, 1], dtype=np.uint8),  # Test byte/boolean
            'comments': np.array(["This is star 1", "This is star 2", "This is star 3"], dtype='U20') #String
        }
        table = Table(data)
        hdu = fits.BinTableHDU(table, name="MYTABLE")
        hdu.writeto(cls.table_file, overwrite=True)

        # --- MEF File ---
        cls.mef_file = os.path.join(cls.test_dir, "test_mef.fits")
        primary_hdu = fits.PrimaryHDU()  # Empty primary
        ext1 = fits.ImageHDU(np.arange(100, dtype=np.float32).reshape(10, 10), name='EXT1')
        ext2 = fits.ImageHDU(np.arange(100, 200, dtype=np.float32).reshape(10, 10), name='EXT2')
        ext3 = fits.BinTableHDU(table, name="TABLE_EXT")

        ext1.header['CTYPE1'] = 'RA---TAN'
        ext1.header['CTYPE2'] = 'DEC--TAN'
        ext1.header['CRVAL1'] = 202.5
        ext1.header['CRVAL2'] = 47.5
        ext1.header['CRPIX1'] = 5.0
        ext1.header['CRPIX2'] = 5.0
        ext1.header['CDELT1'] = -0.001
        ext1.header['CDELT2'] = 0.001

        ext2.header['CTYPE1'] = 'RA---TAN'
        ext2.header['CTYPE2'] = 'DEC--TAN'
        ext2.header['CRVAL1'] = 102.5
        ext2.header['CRVAL2'] = 27.5
        ext2.header['CRPIX1'] = 5.0
        ext2.header['CRPIX2'] = 5.0
        ext2.header['CDELT1'] = -0.002
        ext2.header['CDELT2'] = 0.002
        hdul = fits.HDUList([primary_hdu, ext1, ext2, ext3])
        hdul.writeto(cls.mef_file, overwrite=True)

    @classmethod
    def tearDownClass(cls):
        # Clean up the test files.
        os.remove(cls.image_file)
        os.remove(cls.cube_file)
        os.remove(cls.table_file)
        os.remove(cls.mef_file)
        os.remove(cls.image_1d_file)
        os.rmdir(cls.test_dir)

    def test_read_full_image(self):
        data, header = torchfits.read(self.image_file)
        self.assertTrue(isinstance(data, torch.Tensor))
        self.assertEqual(data.dtype, torch.float32)
        self.assertEqual(data.shape, (10, 10))
        self.assertTrue(np.allclose(data.numpy(), self.image_data))
        self.assertEqual(header['CTYPE1'], 'RA---TAN')
        self.assertEqual(header['OBJECT'], 'Test Image')

        # Test reading with HDU number
        data2, _ = torchfits.read(self.image_file, hdu=1)
        self.assertTrue(np.allclose(data.numpy(), data2.numpy()))

    def test_read_image_cutout_string(self):
        data, _ = torchfits.read(f"{self.image_file}[1][2:5,3:7]")  # Use extension number
        self.assertEqual(data.shape, (3, 4))
        self.assertTrue(np.allclose(data.numpy(), self.image_data[2:5, 3:7]))

    def test_read_image_region(self):
        data, _ = torchfits.read(self.image_file, hdu=1, start=[2, 3], shape=[3, 4])
        self.assertEqual(data.shape, (3, 4))
        self.assertTrue(np.allclose(data.numpy(), self.image_data[2:5, 3:7]))

    def test_read_image_1d(self):
        data, _ = torchfits.read(self.image_1d_file)
        self.assertEqual(data.ndim, 1)
        self.assertTrue(np.allclose(data.numpy(), self.image_1d_data))

    def test_read_image_region_none_shape(self):
        # Test reading to the end of a dimension using shape=None equivalent
        data, _ = torchfits.read(self.image_file, hdu=1, start=[2,3], shape=[3, -1])
        self.assertEqual(data.shape, (3, 7))
        self.assertTrue(np.allclose(data.numpy(), self.image_data[2:5, 3:]))

    def test_read_cube_full(self):
        data, _ = torchfits.read(self.cube_file)
        self.assertEqual(data.dtype, torch.float64)
        self.assertEqual(data.shape, (2, 3, 4))
        self.assertTrue(np.allclose(data.numpy(), self.cube_data))

    def test_read_cube_region(self):
        data, _ = torchfits.read(self.cube_file, hdu=1, start=[0, 1, 2], shape=[1, 2, 1])
        self.assertEqual(data.shape, (1, 2, 1))
        self.assertTrue(np.allclose(data.numpy(), self.cube_data[0:1, 1:3, 2:3]))

    def test_read_cube_slice_none(self):
        # Test reading a 2D slice with None for shape
        data, _ = torchfits.read(self.cube_file, hdu=1, start=[0,0,1], shape=[-1,-1,1])
        self.assertEqual(data.shape, (2,3,1))
        self.assertTrue(np.allclose(data.numpy(), self.cube_data[:,:,1:2]))

        data, _ = torchfits.read(self.cube_file, hdu=1, start=[0,1,0], shape=[-1,1,-1])
        self.assertEqual(data.shape, (2,1,4))
        self.assertTrue(np.allclose(data.numpy(), self.cube_data[:,1:2,:]))

    def test_read_table(self):
        table_data = torchfits.read(self.table_file, hdu="MYTABLE")  # Use extension name
        self.assertTrue(isinstance(table_data, dict))
        self.assertEqual(set(table_data.keys()), {'ra', 'dec', 'flux', 'id', 'flag', 'comments'})
        self.assertEqual(table_data['ra'].dtype, torch.float64)
        self.assertEqual(table_data['dec'].dtype, torch.float64)
        self.assertEqual(table_data['flux'].dtype, torch.float32)
        self.assertEqual(table_data['id'].dtype, torch.int32)
        self.assertEqual(table_data['flag'].dtype, torch.uint8)
        self.assertTrue(np.allclose(table_data['ra'].numpy(), [200.0, 201.0, 202.0]))
         # Check string
        self.assertTrue(isinstance(table_data['comments'], list))
        self.assertEqual(table_data['comments'], ['This is star 1', 'This is star 2', 'This is star 3'])


    def test_get_header(self):
        header = torchfits.get_header(self.image_file, 1)
        self.assertEqual(header['CTYPE1'], 'RA---TAN')
        self.assertEqual(header['OBJECT'], 'Test Image')
        header = torchfits.get_header(self.table_file, "MYTABLE") # Test with named HDU
        self.assertTrue('TFORM1' in header)


    def test_get_dims(self):
        dims = torchfits.get_dims(self.image_file, 1)
        self.assertEqual(dims, [10, 10])

        dims_cube = torchfits.get_dims(self.cube_file, 1)
        self.assertEqual(dims_cube, [4, 3, 2])

    def test_get_header_value(self):
        value = torchfits.get_header_value(self.image_file, 1, "CRVAL1")
        self.assertEqual(value, '202.5')
        value = torchfits.get_header_value(self.image_file, 1, "MISSING")
        self.assertEqual(value, "")

    def test_get_hdu_type(self):
        self.assertEqual(torchfits.get_hdu_type(self.image_file, 1), "IMAGE")
        self.assertEqual(torchfits.get_hdu_type(self.table_file, 1), "BINTABLE")

    def test_get_num_hdus(self):
        self.assertEqual(torchfits.get_num_hdus(self.image_file), 1)
        self.assertEqual(torchfits.get_num_hdus(self.mef_file), 4)

    def test_errors(self):
        with self.assertRaises(RuntimeError):
            torchfits.read("nonexistent.fits")
        with self.assertRaises(RuntimeError):
            torchfits.read_region(self.image_file, 1, [0, 0], [11, 11])  # Out of bounds
        with self.assertRaises(RuntimeError):
            torchfits.read(self.image_file, hdu=2)  # Invalid HDU
        with self.assertRaises(RuntimeError):
            torchfits.read_region(self.image_file, 1, [0], [1, 2]) #Dimension mismatch
        with self.assertRaises(RuntimeError):
            torchfits.read_region(self.image_file, 1, [0,0], [0, 2])  # Invalid shape
        with self.assertRaises(RuntimeError):
            torchfits.get_header_value(self.image_file, 1, "  ") #Invalid key
        with self.assertRaises(RuntimeError):
            torchfits.read(self.image_file, hdu=1, start=[0, 0])  # Missing shape
        with self.assertRaises(RuntimeError):
            torchfits.read(self.image_file, hdu=1,  shape=[1, 2])  # Missing start
        with self.assertRaises(RuntimeError):
            torchfits.read(self.image_file, hdu=1.5)  # Bad hdu type
        with self.assertRaises(RuntimeError):
            torchfits.read(self.image_file, hdu=1, start=1, shape=[1,2]) #Bad start type
        with self.assertRaises(RuntimeError):
            torchfits.read(self.image_file, hdu=1, start=[0,1], shape=1) #Bad shape type

    def test_mef_iteration(self):
        # Test iterating through HDUs
        num_hdus = torchfits.get_num_hdus(self.mef_file)
        self.assertEqual(num_hdus, 4)

        for hdu_num in range(1, num_hdus + 1):
            try:
                hdu_type = torchfits.get_hdu_type(self.mef_file, hdu_num)
                header = torchfits.get_header(self.mef_file, hdu_num)
                print(f"HDU {hdu_num}: Type = {hdu_type}, EXTNAME = {header.get('EXTNAME', 'N/A')}")
                if hdu_type == "IMAGE":
                    data, _ = torchfits.read(self.mef_file, hdu=hdu_num)
                    self.assertTrue(isinstance(data, torch.Tensor))
                elif hdu_type == "BINTABLE":
                    table = torchfits.read(self.mef_file, hdu=hdu_num)
                    self.assertTrue(isinstance(table, dict))

            except RuntimeError as e:
                self.fail(f"Error reading HDU {hdu_num}: {e}")

    def test_read_with_cfitsio_string_and_hdu(self):
        # Test that providing hdu is ignored when CFITSIO string includes extension
        data1, _ = torchfits.read(f"{self.image_file}[1][2:5,3:7]", hdu=2)  # hdu=2 should be ignored
        data2, _ = torchfits.read(f"{self.image_file}[1][2:5,3:7]") #Correct one
        self.assertTrue(np.allclose(data1.numpy(), data2.numpy()))

        #Test named HDU
        data1, _ = torchfits.read(f"{self.mef_file}[EXT1][2:5,3:7]")  # Use extension name
        data2, _ = torchfits.read(f"{self.mef_file}[2][2:5,3:7]") #Use extension number
        self.assertTrue(np.allclose(data1.numpy(), data2.numpy()))

        data3, _ = torchfits.read(f"{self.mef_file}[EXT1]", hdu=2)  # hdu=2 should be ignored
        data4, _ = torchfits.read(f"{self.mef_file}[EXT1]")
        self.assertTrue(np.allclose(data3.numpy(), data4.numpy()))

    def test_read_spectrum_2d(self):
        # From 2D image.  Extract a row and a column.
        row_data, _ = torchfits.read(self.image_file, hdu=1, start=[2, 0], shape=[1, -1])
        self.assertEqual(row_data.shape, (1, 10))
        self.assertTrue(np.allclose(row_data.numpy(), self.image_data[2:3, :]))

        col_data, _ = torchfits.read(self.image_file, hdu=1, start=[0, 5], shape=[-1, 1])
        self.assertEqual(col_data.shape, (10, 1))
        self.assertTrue(np.allclose(col_data.numpy(), self.image_data[:, 5:6]))


    def test_read_spectrum_3d(self):
        # From 3D cube. Extract spectra along different axes.
        spec_z, _ = torchfits.read(self.cube_file, hdu=1, start=[1, 2, 0], shape=[1, 1, -1])  # x, y, z
        self.assertEqual(spec_z.shape, (1, 1, 2))
        self.assertTrue(np.allclose(spec_z.numpy(), self.cube_data[0:1, 2:3, :]))

        spec_y, _ = torchfits.read(self.cube_file, hdu=1, start=[1, 0, 1], shape=[1, -1, 1])  # x, y, z
        self.assertEqual(spec_y.shape, (1, 3, 1))
        self.assertTrue(np.allclose(spec_y.numpy(), self.cube_data[0:1, :, 1:2]))

        spec_x, _ = torchfits.read(self.cube_file, hdu=1, start=[0, 1, 1], shape=[-1, 1, 1])  # x, y, z
        self.assertEqual(spec_x.shape, (2, 1, 1))
        self.assertTrue(np.allclose(spec_x.numpy(), self.cube_data[:, 1:2, 1:2]))

    def test_read_spectrum_1d(self):
        #From a 1D image
        data, _ = torchfits.read(self.image_1d_file)
        self.assertEqual(data.ndim, 1)
        self.assertTrue(np.allclose(data.numpy(), self.image_1d_data))

    def test_empty_primary(self):
        #Test read an empty primary HDU
        data, header = torchfits.read(self.mef_file)
        self.assertIsNone(data)

if __name__ == '__main__':
    unittest.main()
    