"""
Tests for torchfits dataloader module.
"""

from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils.data import RandomSampler

from torchfits.dataloader import (
    create_dataloader,
    create_distributed_dataloader,
    create_fits_dataloader,
    create_ml_dataloader,
    create_streaming_dataloader,
    create_table_dataloader,
    get_optimal_dataloader_config,
)
from torchfits.datasets import FITSDataset, IterableFITSDataset


class TestDataLoaderCreation:
    """Test DataLoader creation functions."""

    def test_create_dataloader_with_dataset(self):
        """Test creating DataLoader with FITSDataset."""
        # Mock dataset
        mock_dataset = Mock(spec=FITSDataset)
        mock_dataset.__len__ = Mock(return_value=100)

        dataloader = create_dataloader(
            mock_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing in tests
        )

        assert dataloader.batch_size == 16
        assert isinstance(dataloader.sampler, RandomSampler)
        assert dataloader.num_workers == 0

    def test_create_dataloader_with_file_list(self):
        """Test creating DataLoader with list of file paths."""
        file_paths = ["file1.fits", "file2.fits", "file3.fits"]

        with patch("torchfits.dataloader.FITSDataset") as mock_dataset_class:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=3)
            mock_dataset_class.return_value = mock_dataset

            create_dataloader(file_paths, batch_size=2, num_workers=0)
            # Should have created FITSDataset
            mock_dataset_class.assert_called_once_with(file_paths)

    def test_create_dataloader_with_iterable_dataset(self):
        """Test creating DataLoader with IterableFITSDataset."""
        mock_dataset = Mock(spec=IterableFITSDataset)

        dataloader = create_dataloader(
            mock_dataset,
            batch_size=8,
            shuffle=True,  # Should be ignored for IterableDataset
            num_workers=0,
        )

        assert dataloader.batch_size == 8
        # IterableDataset uses _InfiniteConstantSampler internally in newer PyTorch versions
        # so checking for None is fragile.
        # assert dataloader.sampler is None

    def test_create_fits_dataloader(self):
        """Test convenience function for FITS DataLoader."""
        file_paths = ["file1.fits", "file2.fits"]

        with patch("torchfits.dataloader.FITSDataset") as mock_dataset_class:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=2)
            mock_dataset_class.return_value = mock_dataset

            create_fits_dataloader(file_paths, hdu=1, batch_size=4, num_workers=0)

            mock_dataset_class.assert_called_once_with(
                file_paths=file_paths,
                hdu=1,
                transform=None,
                device="cpu",
                include_header=False,
            )

    def test_create_streaming_dataloader(self):
        """Test streaming DataLoader creation."""
        with patch("torchfits.dataloader.IterableFITSDataset") as mock_dataset_class:
            mock_dataset = Mock(spec=IterableFITSDataset)
            mock_dataset_class.return_value = mock_dataset

            create_streaming_dataloader(
                index_url="http://example.com/index.json",
                hdu=0,
                batch_size=16,
                num_workers=0,
            )

    def test_create_table_dataloader(self):
        """Test table chunk DataLoader creation."""
        file_paths = ["file1.fits"]

        with patch("torchfits.dataloader.TableChunkDataset") as mock_dataset_class:
            mock_dataset = Mock()
            mock_dataset_class.return_value = mock_dataset

            create_table_dataloader(file_paths, hdu=1, chunk_rows=100, num_workers=0)

            mock_dataset_class.assert_called_once_with(
                file_paths=file_paths,
                hdu=1,
                columns=None,
                chunk_rows=100,
                max_chunks=None,
                mmap=False,
                device="cpu",
                non_blocking_transfer=True,
                pin_memory_transfer=False,
                transform=None,
                include_header=False,
            )


class TestDistributedDataLoader:
    """Test distributed DataLoader functionality."""

    def test_create_distributed_dataloader(self):
        """Test distributed DataLoader creation."""
        file_paths = ["file1.fits", "file2.fits", "file3.fits", "file4.fits"]

        with (
            patch("torchfits.dataloader.FITSDataset") as mock_dataset_class,
            patch("torchfits.dataloader.DistributedSampler") as mock_sampler_class,
        ):
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=4)
            mock_dataset_class.return_value = mock_dataset

            mock_sampler = Mock()
            mock_sampler_class.return_value = mock_sampler

            create_distributed_dataloader(
                file_paths,
                batch_size=2,
                num_replicas=2,
                rank=0,
                shuffle=True,
                seed=42,
                num_workers=0,
            )

            # Should create DistributedSampler
            mock_sampler_class.assert_called_once_with(
                mock_dataset, num_replicas=2, rank=0, shuffle=True, seed=42
            )


class TestMLDataLoader:
    """Test ML-specific DataLoader functionality."""

    def test_create_ml_dataloader(self):
        """Test ML DataLoader with train/val/test splits."""
        file_paths = [f"file{i}.fits" for i in range(100)]

        with patch("torchfits.dataloader.create_fits_dataloader") as mock_create:
            mock_dataloader = Mock()
            mock_create.return_value = mock_dataloader

            dataloaders = create_ml_dataloader(
                file_paths,
                batch_size=16,
                validation_split=0.2,
                test_split=0.1,
                random_seed=42,
            )

            # Should create three dataloaders
            assert "train" in dataloaders
            assert "val" in dataloaders
            assert "test" in dataloaders

            # Should have called create_fits_dataloader 3 times
            assert mock_create.call_count == 3

    def test_ml_dataloader_with_transforms(self):
        """Test ML DataLoader with custom transforms."""
        file_paths = [f"file{i}.fits" for i in range(50)]

        mock_train_transform = Mock()
        mock_val_transform = Mock()

        transforms = {"train": mock_train_transform, "val": mock_val_transform}

        with patch("torchfits.dataloader.create_fits_dataloader") as mock_create:
            mock_dataloader = Mock()
            mock_create.return_value = mock_dataloader

            dataloaders = create_ml_dataloader(
                file_paths,
                transforms=transforms,
                validation_split=0.3,
                test_split=0.0,  # No test split
            )

            # Should only create train and val dataloaders
            assert "train" in dataloaders
            assert "val" in dataloaders
            assert "test" not in dataloaders


class TestOptimalConfiguration:
    """Test optimal DataLoader configuration."""

    def test_get_optimal_dataloader_config(self):
        """Test optimal configuration calculation."""
        config = get_optimal_dataloader_config(
            dataset_size=1000, file_size_mb=10.0, available_memory_gb=8.0, num_gpus=1
        )

        assert "batch_size" in config
        assert "num_workers" in config
        assert "pin_memory" in config
        assert "prefetch_factor" in config
        assert "persistent_workers" in config

        assert config["batch_size"] > 0
        assert config["num_workers"] >= 0
        assert isinstance(config["pin_memory"], bool)

    def test_optimal_config_small_dataset(self):
        """Test configuration for small datasets."""
        config = get_optimal_dataloader_config(
            dataset_size=100, file_size_mb=5.0, available_memory_gb=4.0, num_gpus=0
        )

        # Small dataset should have conservative settings
        assert config["num_workers"] <= 2
        assert config["batch_size"] <= 16
        assert not config["pin_memory"]  # No GPU

    def test_optimal_config_large_dataset(self):
        """Test configuration for large datasets."""
        config = get_optimal_dataloader_config(
            dataset_size=100000, file_size_mb=50.0, available_memory_gb=32.0, num_gpus=2
        )

        # Large dataset with GPUs should have aggressive settings
        assert config["num_workers"] > 2
        assert config["pin_memory"]  # Has GPUs
        assert config["persistent_workers"]


class TestDataLoaderParameters:
    """Test DataLoader parameter handling."""

    def test_pin_memory_with_cuda(self):
        """Test pin_memory setting with CUDA availability."""
        file_paths = ["file1.fits"]

        with (
            patch("torchfits.dataloader.FITSDataset") as mock_dataset_class,
            patch("torch.cuda.is_available", return_value=True),
        ):
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=1)
            mock_dataset_class.return_value = mock_dataset

            dataloader = create_dataloader(file_paths, pin_memory=True, num_workers=0)

            assert dataloader.pin_memory

    def test_pin_memory_without_cuda(self):
        """Test pin_memory setting without CUDA."""
        file_paths = ["file1.fits"]

        with (
            patch("torchfits.dataloader.FITSDataset") as mock_dataset_class,
            patch("torch.cuda.is_available", return_value=False),
        ):
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=1)
            mock_dataset_class.return_value = mock_dataset

            dataloader = create_dataloader(
                file_paths,
                pin_memory=True,
                num_workers=0,  # Requested True
            )

            assert not dataloader.pin_memory  # Should be False without CUDA

    def test_prefetch_factor_with_workers(self):
        """Test prefetch_factor with multiple workers."""
        file_paths = ["file1.fits"]

        with patch("torchfits.dataloader.FITSDataset") as mock_dataset_class:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=1)
            mock_dataset_class.return_value = mock_dataset

            dataloader = create_dataloader(file_paths, num_workers=2, prefetch_factor=4)

            assert dataloader.prefetch_factor == 4

    def test_prefetch_factor_without_workers(self):
        """Test prefetch_factor with no workers."""
        file_paths = ["file1.fits"]

        with patch("torchfits.dataloader.FITSDataset") as mock_dataset_class:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=1)
            mock_dataset_class.return_value = mock_dataset

            dataloader = create_dataloader(file_paths, num_workers=0)
            assert dataloader.prefetch_factor is None  # Default for no workers


@pytest.mark.integration
class TestDataLoaderIntegration:
    """Integration tests for DataLoader functionality."""

    def test_dataloader_iteration(self):
        """Test that DataLoader can be iterated (with mock data)."""
        # Create mock dataset that returns tensors
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=4)
        mock_dataset.__getitem__ = Mock(
            side_effect=[
                torch.randn(10, 10),
                torch.randn(10, 10),
                torch.randn(10, 10),
                torch.randn(10, 10),
            ]
        )

        dataloader = create_dataloader(
            mock_dataset, batch_size=2, shuffle=False, num_workers=0
        )

        batches = list(dataloader)

        assert len(batches) == 2  # 4 samples / batch_size 2
        assert all(batch.shape[0] == 2 for batch in batches)  # Batch size
        assert all(batch.shape[1:] == (10, 10) for batch in batches)  # Sample shape
