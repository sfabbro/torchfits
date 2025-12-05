#!/usr/bin/env python3
"""
Machine Learning pipeline example with torchfits.
Demonstrates dataset creation, transforms, and training workflows.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path for examples
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from astropy.io import fits

import torchfits
from torchfits.transforms import (
    ZScale,
    RandomCrop,
    RandomFlip,
    GaussianNoise,
    Compose,
    create_training_transform,
    create_validation_transform,
)
from torchfits.datasets import FITSDataset
from torchfits.dataloader import create_fits_dataloader


def create_synthetic_dataset(num_files=100, image_size=(128, 128)):
    """Create synthetic astronomical dataset."""
    print(f"Creating {num_files} synthetic FITS files...")

    file_paths = []
    labels = []

    for i in range(num_files):
        # Create realistic astronomical image
        data = np.random.normal(100, 10, image_size).astype(np.float32)

        # Add sources based on class
        class_id = i % 3  # 3 classes: 0=empty, 1=single source, 2=multiple sources

        if class_id == 1:  # Single bright source
            y, x = image_size[0] // 2, image_size[1] // 2
            yy, xx = np.ogrid[: image_size[0], : image_size[1]]
            source = 1000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 10**2))
            data += source

        elif class_id == 2:  # Multiple sources
            for _ in range(3):
                y = np.random.randint(20, image_size[0] - 20)
                x = np.random.randint(20, image_size[1] - 20)
                yy, xx = np.ogrid[: image_size[0], : image_size[1]]
                source = 500 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 5**2))
                data += source

        # Save as FITS file
        filename = f"synthetic_{i:03d}.fits"
        hdu = fits.PrimaryHDU(data)
        hdu.header["CLASS"] = class_id
        hdu.header["OBJECT"] = f"SyntheticObject_{i}"
        hdu.writeto(filename, overwrite=True)

        file_paths.append(filename)
        labels.append(class_id)

    return file_paths, labels


class SimpleClassifier(nn.Module):
    """Simple CNN for astronomical image classification."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def demo_dataset_creation():
    """Demonstrate FITS dataset creation."""
    print("📦 Dataset Creation")
    print("-" * 20)

    # Create synthetic data
    file_paths, labels = create_synthetic_dataset(50, (64, 64))

    try:
        # Create training transforms
        train_transform = create_training_transform(
            crop_size=56, normalize=True, augment=True
        )

        val_transform = create_validation_transform(crop_size=56, normalize=True)

        # Create datasets
        train_files = file_paths[:40]
        val_files = file_paths[40:]
        train_labels = labels[:40]
        val_labels = labels[40:]

        # Custom dataset class that includes labels
        class LabeledFITSDataset(FITSDataset):
            def __init__(self, file_paths, labels, **kwargs):
                super().__init__(file_paths, **kwargs)
                self.labels = labels

            def __getitem__(self, idx):
                image = super().__getitem__(idx)
                label = self.labels[idx]
                return image, torch.tensor(label, dtype=torch.long)

        train_dataset = LabeledFITSDataset(
            train_files, train_labels, transform=train_transform
        )
        val_dataset = LabeledFITSDataset(val_files, val_labels, transform=val_transform)

        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")

        # Test dataset access
        sample_image, sample_label = train_dataset[0]
        print(f"  Sample shape: {sample_image.shape}")
        print(f"  Sample dtype: {sample_image.dtype}")
        print(f"  Sample label: {sample_label}")

        return train_dataset, val_dataset, file_paths

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None, None, file_paths


def demo_dataloader_creation(train_dataset, val_dataset):
    """Demonstrate DataLoader creation."""
    print("🔄 DataLoader Creation")
    print("-" * 20)

    try:
        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=8, shuffle=True, num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=8, shuffle=False, num_workers=2
        )

        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")

        # Test batch loading
        batch_images, batch_labels = next(iter(train_loader))
        print(f"  Batch shape: {batch_images.shape}")
        print(f"  Batch labels: {batch_labels}")

        return train_loader, val_loader

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None, None


def demo_training_loop(train_loader, val_loader):
    """Demonstrate training loop."""
    print("🎯 Training Loop")
    print("-" * 20)

    try:
        # Create model
        model = SimpleClassifier(num_classes=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print(f"  Device: {device}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Training loop (just a few epochs for demo)
        num_epochs = 3

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            start_time = time.time()

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # Add channel dimension if needed
                if images.dim() == 3:
                    images = images.unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                if batch_idx == 0:  # Just show first batch info
                    print(f"    Batch 0 - Loss: {loss.item():.4f}")

            epoch_time = time.time() - start_time
            train_acc = 100.0 * train_correct / train_total

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    if images.dim() == 3:
                        images = images.unsqueeze(1)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_acc = 100.0 * val_correct / val_total

            print(f"  Epoch {epoch+1}/{num_epochs}:")
            print(
                f"    Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%"
            )
            print(f"    Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")
            print(f"    Time: {epoch_time:.2f}s")

        print("  ✅ Training completed successfully")

    except Exception as e:
        print(f"  ❌ Error: {e}")


def demo_batch_processing():
    """Demonstrate batch file processing."""
    print("📊 Batch Processing")
    print("-" * 20)

    # Create a few test files
    test_files = []
    try:
        for i in range(5):
            data = np.random.normal(0, 1, (32, 32)).astype(np.float32)
            filename = f"batch_test_{i}.fits"
            hdu = fits.PrimaryHDU(data)
            hdu.writeto(filename, overwrite=True)
            test_files.append(filename)

        # Batch reading
        start_time = time.time()
        batch_results = torchfits.read_batch(test_files, max_workers=4)
        batch_time = time.time() - start_time

        print(f"  Files processed: {len(batch_results)}")
        print(f"  Batch time: {batch_time:.3f}s")

        # Batch info
        info = torchfits.get_batch_info(test_files)
        print(f"  Total size: {info.get('total_size_mb', 0):.2f} MB")
        print(f"  Valid files: {info.get('valid_files', 0)}/{info.get('num_files', 0)}")

        print("  ✅ Batch processing completed")

    except Exception as e:
        print(f"  ❌ Error: {e}")
    finally:
        # Clean up
        for f in test_files:
            if os.path.exists(f):
                os.unlink(f)


def main():
    """Run ML pipeline demonstration."""
    print("🤖 TorchFITS ML Pipeline Example")
    print("=" * 40)
    print("Demonstrating end-to-end machine learning workflow:")
    print("- Synthetic dataset creation")
    print("- FITS dataset and DataLoader setup")
    print("- Training loop with transforms")
    print("- Batch processing utilities")
    print()

    file_paths = []

    try:
        # Dataset creation
        train_dataset, val_dataset, file_paths = demo_dataset_creation()
        print()

        if train_dataset is not None and val_dataset is not None:
            # DataLoader creation
            train_loader, val_loader = demo_dataloader_creation(
                train_dataset, val_dataset
            )
            print()

            if train_loader is not None and val_loader is not None:
                # Training demonstration
                demo_training_loop(train_loader, val_loader)
                print()

        # Batch processing
        demo_batch_processing()
        print()

        print("🎉 ML pipeline demonstration completed successfully!")
        print("This shows how torchfits integrates seamlessly with PyTorch")
        print("for astronomical machine learning workflows.")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1

    finally:
        # Clean up synthetic files
        for f in file_paths:
            if os.path.exists(f):
                os.unlink(f)

    return 0


if __name__ == "__main__":
    sys.exit(main())
