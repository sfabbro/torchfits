import torch
import torchfits
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits

# --- Synthetic Data Generation (for demonstration purposes) ---

def create_dummy_fits(data_dir, num_files=10, size=(64, 64)):
    """Creates a set of dummy FITS files for the example."""
    os.makedirs(data_dir, exist_ok=True)
    for i in range(num_files):
        data = np.random.rand(*size).astype(np.float32)  # Random image data
        hdu = fits.PrimaryHDU(data)
        hdu.header['LABEL'] = i % 2  # Simple binary labels: 0 or 1
        filename = os.path.join(data_dir, f"image_{i:03d}.fits")
        hdu.writeto(filename, overwrite=True)

# --- PyTorch Dataset ---

class SimpleFitsDataset(Dataset):
    def __init__(self, data_dir, cache_capacity=0, device='cpu'):  # Add cache_capacity and device
        self.data_dir = data_dir
        self.file_list = []
        self.labels = []
        self.cache_capacity = cache_capacity
        self.device = device


        # Find all FITS files in the directory and extract labels
        for filename in os.listdir(data_dir):
            if filename.endswith(".fits"):
                filepath = os.path.join(data_dir, filename)
                self.file_list.append(filepath)
                # Get label from header (more robust than filename parsing)
                try:
                    header = torchfits.get_header(filepath, 1) # Get header of primary HDU
                    label = int(header['LABEL'])
                    self.labels.append(label)
                except (RuntimeError, KeyError) as e: #Catch errors
                    print(f"Warning: Skipping file {filename} due to error: {e}")
                    # In a real dataset, you'd probably have a better way
                    # to handle missing/corrupted files.
                    continue #Skip if error.

        #Sort files and labels (for reproducibility)
        self.file_list, self.labels = zip(*sorted(zip(self.file_list, self.labels)))
        self.file_list = list(self.file_list)
        self.labels = list(self.labels)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        label = self.labels[idx]

        try:
            data, _ = torchfits.read(filename, cache_capacity=self.cache_capacity, device=self.device)  # Pass cache and device
            # Add a channel dimension if it's a 2D image (for consistency)
            if data.ndim == 2:
                data = data.unsqueeze(0)  # [H, W] -> [1, H, W]

            return data, torch.tensor(label, dtype=torch.long)

        except RuntimeError as e:
            print(f"Error reading {filename}: {e}")
            return None  # Return None if there's an error

# --- Collate Function (to handle potential None values) ---
def collate_fn(batch):
    # Remove any None values from the batch (caused by read errors)
    batch = [item for item in batch if item is not None]
    if not batch:  # Handle the case where *all* items in a batch are None
        return torch.Tensor(), torch.Tensor()
    return torch.utils.data.default_collate(batch)

# --- Main Script ---

def main():
    data_dir = "data_simple_fits"
    create_dummy_fits(data_dir)  # Generate the synthetic data

    # --- Demonstrate different cache capacities and devices ---
    for capacity in [0, 10]:  # Test with and without caching
        for device in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
            print(f"\n--- Cache Capacity: {capacity}, Device: {device} ---")

            # Create dataset and dataloader
            dataset = SimpleFitsDataset(data_dir, cache_capacity=capacity, device=device)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=(device=='cuda'))

            # Iterate through a few batches
            print("Iterating through DataLoader:")
            for i, (images, labels) in enumerate(dataloader):
                #The tensors are already in the correct device, because of torchfits.
                if images.numel() == 0:
                    continue
                print(f"  Batch {i}:")
                print(f"    Image shape: {images.shape}, Device: {images.device}")  # Show device
                print(f"    Labels: {labels}, Device: {labels.device}")
                if i == 2:
                    break

if __name__ == "__main__":
    main()