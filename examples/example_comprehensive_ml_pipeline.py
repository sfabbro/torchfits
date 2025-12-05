"""
Comprehensive ML Pipeline Example for TorchFITS 0.9

This example demonstrates a complete machine learning pipeline using all
major torchfits features including data loading, transforms, caching,
buffer management, and distributed training support.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import torchfits components
import torchfits
from torchfits import buffer, cache, dataloader, transforms
from torchfits.datasets import FITSDataset


class AstronomicalCNN(nn.Module):
    """Simple CNN for astronomical image classification."""

    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Global average pooling
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MockFITSDataset:
    """Mock FITS dataset for demonstration purposes."""

    def __init__(self, size: int = 1000, image_size: int = 256, num_classes: int = 10):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes

        # Generate synthetic astronomical-like data
        np.random.seed(42)  # For reproducibility
        self.data = []
        self.labels = []

        for i in range(size):
            # Create synthetic astronomical image
            # Background noise
            image = np.random.normal(0, 0.1, (image_size, image_size))

            # Add some "stars" (bright points)
            num_stars = np.random.randint(5, 20)
            for _ in range(num_stars):
                x, y = np.random.randint(10, image_size - 10, 2)
                brightness = np.random.exponential(2.0)
                # Add Gaussian star profile
                xx, yy = np.meshgrid(np.arange(image_size), np.arange(image_size))
                star = brightness * np.exp(
                    -((xx - x) ** 2 + (yy - y) ** 2) / (2 * 3**2)
                )
                image += star

            # Add some "galaxies" (extended sources)
            num_galaxies = np.random.randint(1, 5)
            for _ in range(num_galaxies):
                x, y = np.random.randint(20, image_size - 20, 2)
                brightness = np.random.exponential(1.0)
                size = np.random.uniform(5, 15)
                xx, yy = np.meshgrid(np.arange(image_size), np.arange(image_size))
                galaxy = brightness * np.exp(
                    -((xx - x) ** 2 + (yy - y) ** 2) / (2 * size**2)
                )
                image += galaxy

            self.data.append(torch.tensor(image, dtype=torch.float32))
            self.labels.append(np.random.randint(0, num_classes))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def setup_environment():
    """Set up the torchfits environment with optimal configurations."""
    print("🔧 Setting up TorchFITS environment...")

    # Configure cache for optimal performance
    cache_config = cache.get_optimal_cache_config()
    print(f"  📦 Cache config: {cache_config['environment']} environment detected")
    cache.configure_cache(
        max_files=cache_config["max_files"], max_memory_mb=cache_config["max_memory_mb"]
    )

    # Configure buffers for the expected workload
    buffer_config = buffer.get_optimal_buffer_config(
        {
            "avg_file_size_mb": 10.0,  # Typical astronomical image
            "num_files": 1000,
            "num_workers": 4,
        }
    )
    print(f"  🗄️  Buffer config: {buffer_config['buffer_size_mb']}MB buffers")
    buffer.configure_buffers(
        buffer_size_mb=buffer_config["buffer_size_mb"],
        num_buffers=buffer_config["num_buffers"],
        enable_memory_pool=buffer_config["enable_memory_pool"],
    )

    print("  ✅ Environment setup complete")


def create_transforms():
    """Create training and validation transforms."""
    print("🎨 Creating data transforms...")

    # Training transforms with augmentation
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(224),
            transforms.RandomFlip(horizontal=True, vertical=True, p=0.5),
            transforms.GaussianNoise(std=0.01, snr_based=True),
            transforms.ZScale(),  # Astronomical normalization
            transforms.ToDevice("cuda" if torch.cuda.is_available() else "cpu"),
        ]
    )

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ZScale(),
            transforms.ToDevice("cuda" if torch.cuda.is_available() else "cpu"),
        ]
    )

    print("  ✅ Transforms created")
    return train_transform, val_transform


def create_datasets_and_loaders(train_transform, val_transform):
    """Create datasets and data loaders."""
    print("📊 Creating datasets and data loaders...")

    # Create mock datasets
    train_dataset = MockFITSDataset(size=800, image_size=256, num_classes=5)
    val_dataset = MockFITSDataset(size=200, image_size=256, num_classes=5)

    # Apply transforms
    class TransformDataset:
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            data, label = self.dataset[idx]
            if self.transform:
                data = self.transform(data)
            return data, label

    train_dataset = TransformDataset(train_dataset, train_transform)
    val_dataset = TransformDataset(val_dataset, val_transform)

    # Get optimal DataLoader configuration
    loader_config = dataloader.get_optimal_dataloader_config(
        dataset_size=len(train_dataset),
        file_size_mb=10.0,
        available_memory_gb=16.0,  # Assume 16GB system
        num_gpus=1 if torch.cuda.is_available() else 0,
    )

    print(
        f"  🔄 DataLoader config: batch_size={loader_config['batch_size']}, "
        f"num_workers={loader_config['num_workers']}"
    )

    # Create data loaders
    train_loader = dataloader.create_dataloader(
        train_dataset,
        batch_size=loader_config["batch_size"],
        shuffle=True,
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
        persistent_workers=loader_config["persistent_workers"],
    )

    val_loader = dataloader.create_dataloader(
        val_dataset,
        batch_size=loader_config["batch_size"],
        shuffle=False,
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
        persistent_workers=loader_config["persistent_workers"],
    )

    print("  ✅ Data loaders created")
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, num_epochs: int = 5):
    """Train the model with comprehensive monitoring."""
    print("🚀 Starting model training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)

    # Training metrics
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0

        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            train_samples += data.size(0)

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx:3d}: Loss = {loss.item():.4f}")

        epoch_time = time.time() - start_time
        avg_train_loss = train_loss / train_samples
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)

        val_accuracy = val_correct / val_total
        val_accuracies.append(val_accuracy)

        scheduler.step()

        print(f"  📊 Train Loss: {avg_train_loss:.4f}")
        print(f"  📊 Val Accuracy: {val_accuracy:.4f}")
        print(f"  ⏱️  Epoch Time: {epoch_time:.2f}s")

        # Print system stats
        memory_stats = buffer.get_buffer_stats()
        cache_stats = cache.get_cache_stats()
        print(f"  💾 Buffer Memory: {memory_stats['total_memory_mb']:.1f}MB")
        print(f"  📦 Cache Hit Rate: {cache_stats.get('hit_rate', 0):.2%}")

    print("\n✅ Training completed!")
    return train_losses, val_accuracies


def demonstrate_advanced_features():
    """Demonstrate advanced torchfits features."""
    print("\n🔬 Demonstrating Advanced Features...")

    # 1. Transform performance comparison
    print("\n1️⃣ Transform Performance Comparison")
    test_image = torch.randn(512, 512)

    transform_configs = {
        "ZScale": transforms.ZScale(),
        "AsinhStretch": transforms.AsinhStretch(),
        "LogStretch": transforms.LogStretch(),
        "Compose": transforms.Compose(
            [
                transforms.ZScale(),
                transforms.RandomFlip(),
                transforms.GaussianNoise(std=0.01),
            ]
        ),
    }

    for name, transform in transform_configs.items():
        start_time = time.time()
        for _ in range(10):
            result = transform(test_image)
        avg_time = (time.time() - start_time) / 10
        print(f"  {name:12s}: {avg_time*1000:.2f}ms per transform")

    # 2. Buffer management demonstration
    print("\n2️⃣ Buffer Management Performance")

    # Test memory pool efficiency
    start_time = time.time()
    buffers = []
    for i in range(50):
        buf = buffer.get_buffer_manager().get_buffer(
            f"demo_{i}", (200, 200), torch.float32
        )
        buf.fill_(float(i))
        buffers.append(buf)

    allocation_time = time.time() - start_time
    buffer_stats = buffer.get_buffer_stats()

    print(f"  Buffer allocation: {allocation_time*1000:.2f}ms for 50 buffers")
    print(f"  Memory usage: {buffer_stats['total_memory_mb']:.1f}MB")

    # Clean up
    buffer.clear_buffers()

    # 3. Streaming buffer demonstration
    print("\n3️⃣ Streaming Buffer Performance")

    stream_buf = buffer.create_streaming_buffer(
        "demo_stream", 20, (64, 64), torch.float32
    )

    # Test streaming performance
    test_data = [torch.randn(64, 64) for _ in range(100)]

    start_time = time.time()
    for i, data in enumerate(test_data):
        if i < 20:
            stream_buf.put(data)
        else:
            retrieved = stream_buf.get()
            stream_buf.put(data)

    streaming_time = time.time() - start_time
    throughput = len(test_data) / streaming_time

    print(f"  Streaming throughput: {throughput:.1f} ops/sec")

    # 4. Cache performance
    print("\n4️⃣ Cache Performance")

    cache_stats = cache.get_cache_stats()
    print(f"  Cache configuration: {cache_stats}")

    # Test cache operations
    start_time = time.time()
    for _ in range(1000):
        cache.get_cache_stats()
    stats_time = time.time() - start_time

    print(f"  Cache stats access: {stats_time*1000:.2f}ms for 1000 calls")


def main():
    """Main demonstration function."""
    print("🌟 TorchFITS 0.9 Comprehensive ML Pipeline Demo")
    print("=" * 60)

    try:
        # Setup environment
        setup_environment()

        # Create transforms
        train_transform, val_transform = create_transforms()

        # Create datasets and loaders
        train_loader, val_loader = create_datasets_and_loaders(
            train_transform, val_transform
        )

        # Create model
        print("🧠 Creating model...")
        model = AstronomicalCNN(num_classes=5, input_channels=1)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {total_params:,}")

        # Train model
        train_losses, val_accuracies = train_model(
            model, train_loader, val_loader, num_epochs=3
        )

        # Demonstrate advanced features
        demonstrate_advanced_features()

        # Final statistics
        print("\n📊 Final Statistics:")
        print(f"  Best validation accuracy: {max(val_accuracies):.4f}")
        print(f"  Final training loss: {train_losses[-1]:.4f}")

        # System resource usage
        final_buffer_stats = buffer.get_buffer_stats()
        final_cache_stats = cache.get_cache_stats()

        print(f"  Peak buffer memory: {final_buffer_stats['total_memory_mb']:.1f}MB")
        print(f"  Cache utilization: {final_cache_stats}")

        print("\n🎉 Demo completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        buffer.clear_buffers()
        cache.clear_cache()
        print("  ✅ Cleanup complete")


if __name__ == "__main__":
    main()
