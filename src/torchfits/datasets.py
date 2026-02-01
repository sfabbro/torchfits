"""
PyTorch Dataset classes for FITS data.

This module provides Dataset implementations for machine learning workflows:
- FITSDataset: Map-style dataset for random access
- IterableFITSDataset: Streaming dataset for large-scale data
"""

from typing import Any, Callable, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

# Import will be done locally to avoid circular imports


class FITSDataset(Dataset):
    """
    Map-style dataset for random access to a collection of FITS files.

    Upon initialization, builds a manifest of all possible samples for efficient
    randomization across a vast collection of files.
    """

    def __init__(
        self,
        file_paths: List[str],
        hdu: int = 0,
        transform: Optional[Callable] = None,
        device: str = "cpu",
    ):
        """
        Initialize FITSDataset.

        Args:
            file_paths: List of FITS file paths
            hdu: HDU index to read from each file
            transform: Optional transform to apply to each sample
            device: PyTorch device to load data onto
        """
        self.file_paths = file_paths
        self.hdu = hdu
        self.transform = transform
        self.device = device

        # Build manifest of all samples
        self._build_manifest()

    def _build_manifest(self):
        """Build a manifest of all possible samples."""
        self.manifest = []

        for file_path in self.file_paths:
            # For now, each file is one sample
            # In a full implementation, this could enumerate cutout positions, etc.
            self.manifest.append(
                {
                    "file_path": file_path,
                    "hdu": self.hdu,
                    "sample_id": len(self.manifest),
                }
            )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Any:
        """Get a sample by index."""
        if idx < 0 or idx >= len(self.manifest):
            raise IndexError(f"Index {idx} out of range [0, {len(self.manifest)})")

        sample_info = self.manifest[idx]

        # Load data (import locally to avoid circular imports)
        from . import read

        data = read(
            sample_info["file_path"], hdu=sample_info["hdu"], device=self.device
        )

        # Apply transform if provided
        if self.transform:
            data = self.transform(data)

        return data

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return (
            f"FITSDataset("
            f"num_samples={len(self)}, "
            f"device='{self.device}', "
            f"hdu={self.hdu}"
            f")"
        )


class IterableFITSDataset(IterableDataset):
    """
    Streaming dataset for scenarios where the dataset is too large to list all files.

    Initialized with a URL to a sharded index file. Each worker processes assigned shards
    in a memory-efficient manner.
    """

    def __init__(
        self,
        index_url: str,
        hdu: int = 0,
        transform: Optional[Callable] = None,
        device: str = "cpu",
        shard_size: int = 1000,
    ):
        """
        Initialize IterableFITSDataset.

        Args:
            index_url: URL to sharded index file
            hdu: HDU index to read from each file
            transform: Optional transform to apply to each sample
            device: PyTorch device to load data onto
            shard_size: Number of files per shard
        """
        self.index_url = index_url
        self.hdu = hdu
        self.transform = transform
        self.device = device
        self.shard_size = shard_size

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the dataset."""
        # Get worker info for distributed processing
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single process
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Load and process assigned shards
        for shard_id in self._get_assigned_shards(worker_id, num_workers):
            for sample in self._process_shard(shard_id):
                yield sample

    def _get_assigned_shards(self, worker_id: int, num_workers: int) -> List[int]:
        """Get shard IDs assigned to this worker."""
        # Simplified implementation - would load from index_url
        total_shards = 10  # Placeholder
        shards_per_worker = total_shards // num_workers
        start_shard = worker_id * shards_per_worker
        end_shard = start_shard + shards_per_worker

        if worker_id == num_workers - 1:
            end_shard = total_shards  # Last worker gets remaining shards

        return list(range(start_shard, end_shard))

    def _process_shard(self, shard_id: int) -> Iterator[Any]:
        """Process a single shard and yield samples."""
        # Simplified implementation - would load shard manifest
        shard_files = [f"file_{shard_id}_{i}.fits" for i in range(self.shard_size)]

        for file_path in shard_files:
            try:
                # Load data (import locally to avoid circular imports)
                from . import read

                data = read(file_path, hdu=self.hdu, device=self.device)

                # Apply transform if provided
                if self.transform:
                    data = self.transform(data)

                yield data
            except (IOError, RuntimeError, ValueError) as e:
                # Log error and continue with next file
                from .logging import logger

                logger.error(f"Failed to process file {file_path}: {str(e)}")
                continue
            except Exception as e:
                from .logging import logger

                logger.critical(f"Unexpected error processing {file_path}: {str(e)}")
                raise

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return (
            f"IterableFITSDataset("
            f"index_url='{self.index_url}', "
            f"device='{self.device}', "
            f"hdu={self.hdu}, "
            f"shard_size={self.shard_size}"
            f")"
        )
