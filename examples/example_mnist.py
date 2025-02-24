import torch
import torchfits
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from astropy.io import fits
from tqdm import tqdm
from torchvision import datasets, transforms  # Use torchvision for MNIST

# --- FITS File Generation (Run Once) ---

def create_mnist_fits(data_dir):
    """Converts the MNIST dataset to FITS images."""
    os.makedirs(data_dir, exist_ok=True)
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())

    for i, (image, label) in enumerate(tqdm(train_dataset, desc="Generating Training FITS")):
        filename = os.path.join(data_dir, f"train_{i:05d}_{label}.fits")
        hdu = fits.PrimaryHDU(image.squeeze().numpy())  # Remove channel dim, convert to NumPy
        hdu.header['LABEL'] = int(label)  # Store label in header
        hdu.writeto(filename, overwrite=True)

    for i, (image, label) in enumerate(tqdm(test_dataset, desc="Generating Testing FITS")):
        filename = os.path.join(data_dir, f"test_{i:05d}_{label}.fits")
        hdu = fits.PrimaryHDU(image.squeeze().numpy())
        hdu.header['LABEL'] = int(label)
        hdu.writeto(filename, overwrite=True)

# --- PyTorch Dataset ---

class MNIST_FITS_Dataset(Dataset):
     def __init__(self, data_dir, train=True, cache_capacity=0, device='cpu'): # Add cache and device
        self.data_dir = data_dir
        self.file_list = []
        self.labels = []
        self.cache_capacity = cache_capacity
        self.device = device # Store device


        prefix = "train" if train else "test"
        for filename in os.listdir(data_dir):
            if filename.startswith(prefix) and filename.endswith(".fits"):
                self.file_list.append(os.path.join(data_dir, filename))
                # Extract label from filename (more robust than header for this example)
                label = int(filename.split("_")[-1].split(".")[0])
                self.labels.append(label)

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
            # Pass cache_capacity and device to read
            data, _ = torchfits.read(filename, cache_capacity=self.cache_capacity, device=self.device)
            # Add a channel dimension if it's a 2D image (for consistency)
            if data.ndim == 2:
                data = data.unsqueeze(0)  # [H, W] -> [1, H, W]
            return data, torch.tensor(label, dtype=torch.long)

        except RuntimeError as e:
            print(f"Error reading or processing {filename}: {e}")
            return None, None  # Return None if there's an error
# --- Model (Simple CNN) ---

class MNIST_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # MNIST is 28x28, two pools -> 7x7
        self.fc2 = nn.Linear(128, 10)  # 10 classes (digits 0-9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Collate Function (to handle potential None values) ---
def collate_fn(batch):
    # Remove any None values from the batch (caused by read errors)
    batch = [item for item in batch if item is not None]
    if not batch:  # Handle the case where *all* items in a batch are None
        return torch.Tensor(), torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)


# --- Main Script ---

def main():
    data_dir = "data_mnist_fits"

    # --- Create FITS files (if they don't exist) ---
    if not os.path.exists(os.path.join(data_dir, "train_00000_0.fits")):  # Check for one file
        create_mnist_fits(data_dir)

    # --- Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Create Datasets and DataLoaders ---
    # Demonstrate with and without cache, and with/without GPU
    train_dataset = MNIST_FITS_Dataset(data_dir, train=True, cache_capacity=100, device=device)
    test_dataset = MNIST_FITS_Dataset(data_dir, train=False, cache_capacity=100, device=device) #Use cache
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=(device.type=='cuda'))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=(device.type=='cuda'))

    # --- Initialize Model, Loss, and Optimizer ---
    model = MNIST_Classifier().to(device)  # Move model to device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training Loop ---
    num_epochs = 3  # Keep it short for the example
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            if inputs.numel() == 0:  # Handle empty batch
                continue
             # Move data to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

    print("Training finished!")

    # --- Evaluation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            if inputs.numel() == 0:  # Handle empty batch
                continue

            # Move data to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the test set: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()