"""
PyTorch Dataset classes for FITS data.

This module provides Dataset implementations for machine learning workflows:
- FITSDataset: Map-style dataset for random access
- IterableFITSDataset: Streaming dataset for large-scale data
"""

from typing import Any, Callable, Iterator, List, Optional, Union

import torch
from torch.utils.data import Dataset, IterableDataset

# Import stream_table lazily in methods to avoid circular imports
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
        hdu: Union[int, str, None] = "auto",
        transform: Optional[Callable] = None,
        device: str = "cpu",
        include_header: bool = False,
        mmap: Union[bool, str] = "auto",
        cache_capacity: int = 0,
        handle_cache_capacity: int = 64,
        scale_on_device: bool = True,
        raw_scale: bool = False,
    ):
        """
        Initialize FITSDataset.

        Args:
            file_paths: List of FITS file paths
            hdu: HDU index/name or `"auto"` to detect first payload HDU
            transform: Optional transform to apply to each sample
            device: PyTorch device to load data onto
            mmap: Memory mapping mode for image reads (`True`, `False`, `'auto'`)
            cache_capacity: Python-side data cache entries for repeated reads
            handle_cache_capacity: Open-handle cache entries for repeated reads
            scale_on_device: Apply FITS scaling in the reader fast path
            raw_scale: Return raw stored values instead of physical scaled values
        """
        self.file_paths = file_paths
        self.hdu = hdu
        self.transform = transform
        self.device = device
        self.include_header = include_header
        self.mmap = mmap
        self.cache_capacity = cache_capacity
        self.handle_cache_capacity = handle_cache_capacity
        self.scale_on_device = scale_on_device
        self.raw_scale = raw_scale

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

        result = read(
            sample_info["file_path"],
            hdu=sample_info["hdu"],
            device=self.device,
            mmap=self.mmap,
            cache_capacity=self.cache_capacity,
            handle_cache_capacity=self.handle_cache_capacity,
            scale_on_device=self.scale_on_device,
            raw_scale=self.raw_scale,
            return_header=self.include_header,
        )

        if self.include_header:
            data, header = result
        else:
            data, header = result, None

        # Apply transform if provided
        if self.transform:
            data = self.transform(data)

        return (data, header) if self.include_header else data


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


class TableChunkDataset(IterableDataset):
    """
    Iterable dataset that yields table chunks from one or more FITS files.
    """

    def __init__(
        self,
        file_paths: List[str],
        hdu: int = 1,
        columns: Optional[List[str]] = None,
        chunk_rows: int = 10000,
        max_chunks: Optional[int] = None,
        mmap: bool = False,
        device: str = "cpu",
        non_blocking_transfer: bool = True,
        pin_memory_transfer: bool = False,
        transform: Optional[Callable] = None,
        include_header: bool = False,
    ):
        self.file_paths = file_paths
        self.hdu = hdu
        self.columns = columns
        self.chunk_rows = chunk_rows
        self.max_chunks = max_chunks
        self.mmap = mmap
        self.device = device
        self.non_blocking_transfer = non_blocking_transfer
        self.pin_memory_transfer = pin_memory_transfer
        self.transform = transform
        self.include_header = include_header

    def __iter__(self) -> Iterator[Any]:
        from . import get_header, table as table_api

        for path in self.file_paths:
            header = get_header(path, self.hdu) if self.include_header else None
            emitted = 0
            for chunk in table_api.scan_torch(
                path,
                hdu=self.hdu,
                columns=self.columns,
                batch_size=self.chunk_rows,
                mmap=self.mmap,
                device=self.device,
                non_blocking=self.non_blocking_transfer,
                pin_memory=self.pin_memory_transfer,
            ):
                if self.transform:
                    chunk = self.transform(chunk)

                yield (chunk, header) if self.include_header else chunk
                emitted += 1
                if self.max_chunks is not None and emitted >= self.max_chunks:
                    break
