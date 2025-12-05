import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from astropy.io import fits
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms  # Use torchvision for MNIST
from tqdm import tqdm

import torchfits

# --- FITS File Generation (Run Once) ---


def create_mnist_fits(data_dir, max_samples=1000):
    """Converts a subset of MNIST dataset to FITS images using torchfits.write().

    Args:
        data_dir: Directory to save FITS files
        max_samples: Maximum number of samples per split (train/test) to convert
    """
    os.makedirs(data_dir, exist_ok=True)

    # Download MNIST using torchvision
    print("Downloading MNIST dataset...")
    train_dataset = datasets.MNIST(
        root=os.path.join(data_dir, "mnist_raw"),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_dataset = datasets.MNIST(
        root=os.path.join(data_dir, "mnist_raw"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    # Convert training samples to FITS using torchfits.write()
    print(f"Converting {max_samples} training samples to FITS...")
    for i in tqdm(range(min(max_samples, len(train_dataset))), desc="Training FITS"):
        image, label = train_dataset[i]
        filename = os.path.join(data_dir, f"train_{i:05d}_{label}.fits")

        # Use torchfits.write() directly with the tensor
        # Remove channel dimension: [1, 28, 28] -> [28, 28]
        image_2d = image.squeeze(0)

        # Create header with label
        header = {"LABEL": int(label), "SPLIT": "TRAIN"}

        # Write using torchfits - showcases direct tensor-to-FITS conversion
        torchfits.write(filename, image_2d, header=header, overwrite=True)

    # Convert test samples to FITS
    print(f"Converting {max_samples} test samples to FITS...")
    for i in tqdm(range(min(max_samples, len(test_dataset))), desc="Testing FITS"):
        image, label = test_dataset[i]
        filename = os.path.join(data_dir, f"test_{i:05d}_{label}.fits")

        image_2d = image.squeeze(0)
        header = {"LABEL": int(label), "SPLIT": "TEST"}

        torchfits.write(filename, image_2d, header=header, overwrite=True)

    print(f"Created {2 * max_samples} FITS files in {data_dir}")


# --- PyTorch Dataset ---


class MNIST_FITS_Dataset(Dataset):
    def __init__(
        self, data_dir, train=True, cache_capacity=0, device="cpu"
    ):  # Add cache and device
        self.data_dir = data_dir
        self.file_list = []
        self.labels = []
        self.cache_capacity = cache_capacity
        self.device = device  # Store device

        prefix = "train" if train else "test"
        for filename in os.listdir(data_dir):
            if filename.startswith(prefix) and filename.endswith(".fits"):
                self.file_list.append(os.path.join(data_dir, filename))
                # Extract label from filename (more robust than header for this example)
                label = int(filename.split("_")[-1].split(".")[0])
                self.labels.append(label)

        # Sort files and labels (for reproducibility)
        self.file_list, self.labels = zip(*sorted(zip(self.file_list, self.labels)))
        self.file_list = list(self.file_list)
        self.labels = list(self.labels)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        label = self.labels[idx]
        try:
            # Pass cache_capacity and device to read (convert device to string)
            device_str = (
                str(self.device) if hasattr(self.device, "__str__") else self.device
            )
            data, _ = torchfits.read(
                filename, cache_capacity=self.cache_capacity, device=device_str
            )
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
    return torch.utils.data.default_collate(batch)


# --- Main Script ---


def main():
    data_dir = "data_mnist_fits"

    # Use a subset for faster demonstration (1000 train, 200 test)
    max_train = 1000
    max_test = 200

    # --- Create FITS files (if they don't exist) ---
    if not os.path.exists(os.path.join(data_dir, "train_00000_5.fits")):
        print("Creating MNIST FITS dataset (subset for demo)...")
        create_mnist_fits(data_dir, max_samples=max_train)
    else:
        print(f"Using existing FITS files in {data_dir}")

    # --- Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # --- Create Datasets and DataLoaders ---
    print("\nCreating datasets...")
    train_dataset = MNIST_FITS_Dataset(
        data_dir, train=True, cache_capacity=100, device=str(device)
    )
    test_dataset = MNIST_FITS_Dataset(
        data_dir, train=False, cache_capacity=100, device=str(device)
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # Use 0 for compatibility
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # --- Initialize Model, Loss, and Optimizer ---
    print("\nInitializing model...")
    model = MNIST_Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training Loop ---
    num_epochs = 5
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if inputs.numel() == 0:
                continue

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Train Acc = {epoch_acc:.2f}%")

    print("\nTraining finished!")

    # --- Evaluation ---
    print("\nEvaluating on test set...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            if inputs.numel() == 0:
                continue

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"\n{'='*50}")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*50}")

    # Show some predictions
    print("\nSample predictions:")
    model.eval()
    with torch.no_grad():
        sample_inputs, sample_labels = next(iter(test_loader))
        if sample_inputs.numel() > 0:
            sample_inputs = sample_inputs[:5].to(device)
            sample_labels = sample_labels[:5]
            outputs = model(sample_inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(sample_labels)):
                print(
                    f"  True: {sample_labels[i].item()}, Predicted: {predicted[i].item()}"
                )


if __name__ == "__main__":
    main()
