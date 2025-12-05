import torchfits
from astropy.io import fits
import os
import torch
import numpy as np

def test_writing():
    filename = "test_write.fits"
    
    # Create TensorHDU
    data_tensor = torch.randn(10, 10)
    header_tensor = torchfits.Header({'TESTKEY': 'TESTVAL'})
    hdu_tensor = torchfits.TensorHDU(data=data_tensor, header=header_tensor)
    
    # Create TableHDU
    data_table = {
        'col1': torch.tensor([1, 2, 3], dtype=torch.int32),
        'col2': torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32)
    }
    header_table = torchfits.Header({'TBLKEY': 'TBLVAL'})
    hdu_table = torchfits.TableHDU(data_table, header=header_table)
    
    # Create HDUList
    hdul = torchfits.HDUList([hdu_tensor, hdu_table])
    
    # Write to file
    hdul.writeto(filename, overwrite=True)
    
    # Verify with Astropy
    try:
        with fits.open(filename) as hdul_astro:
            # Verify TensorHDU (Primary)
            # Handle endianness for PyTorch
            data_numpy = hdul_astro[0].data
            if data_numpy.dtype.byteorder == '>':
                data_numpy = data_numpy.astype(data_numpy.dtype.newbyteorder('<'))
            data_read = torch.tensor(data_numpy)
            assert torch.allclose(data_read, data_tensor)
            assert hdul_astro[0].header['TESTKEY'] == 'TESTVAL'
            
            # Verify TableHDU (Extension 1)
            # Note: Astropy might read as BinTableHDU
            data_read_table = hdul_astro[1].data
            assert np.allclose(data_read_table['col1'], data_table['col1'].numpy())
            assert np.allclose(data_read_table['col2'], data_table['col2'].numpy())
            assert hdul_astro[1].header['TBLKEY'] == 'TBLVAL'
            
    finally:
        if os.path.exists(filename):
            os.remove(filename)

