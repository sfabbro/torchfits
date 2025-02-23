# examples/example_pytorch_dataset.py
import torch
import torchfits
import numpy as np
import os
from astropy.io import fits
from torch.utils.data import Dataset, DataLoader

def create_test_file(filename):
    if not os.path.exists(filename):
        # Create a larger FITS file for the dataset example.
        data = np.random.rand(100, 100).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(filename, overwrite=True)


class FITS_Dataset(Dataset):
    def __init__(self, filename, hdu_num, cutout_size):
        self.filename = filename
        self.hdu_num = hdu_num
        self.cutout_size = cutout_size
        self.dims = torchfits.get_dims(filename, hdu_num) # Get dimensions *once*
        if len(self.dims) != 2:
             raise ValueError("This example Dataset is for 2D images only.")

    def __len__(self):
        # For simplicity, we'll just return a fixed number of samples.
        # In a real application, you'd likely calculate the total number
        # of possible cutouts.
        return 100

    def __getitem__(self, idx):
        # Generate random start coordinates for the cutout.
        start_x = np.random.randint(0, self.dims[0] - self.cutout_size[0])
        start_y = np.random.randint(0, self.dims[1] - self.cutout_size[1])
        start = [start_x, start_y]

        # Read the cutout.
        data, _ = torchfits.read_region(self.filename, self.hdu_num, start, self.cutout_size)
        return data

def main():
    test_file = "dataset_example.fits"
    create_test_file(test_file)
    cutout_size = (32, 32)

    # Create a Dataset.
    dataset = FITS_Dataset(test_file, 1, cutout_size)

    # Create a DataLoader.
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2) #num_workers > 0

    # Iterate through the data.
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: Shape: {batch.shape}, Dtype: {batch.dtype}")
        if i == 2:
          break #Just for demonstration

if __name__ == "__main__":
    main()