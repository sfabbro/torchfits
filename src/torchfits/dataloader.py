"""
DataLoader factory functions for torchfits.

This module provides high-level functions to create optimized DataLoaders
for FITS datasets with appropriate defaults for astronomical data.
"""

from typing import Optional, Callable, Union, List
import torch
from torch.utils.data import DataLoader

from .datasets import FITSDataset, IterableFITSDataset


def create_dataloader(
    dataset: Union[FITSDataset, IterableFITSDataset, List[str]],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    **kwargs
) -> DataLoader:
    """
    Create a high-performance DataLoader for FITS data.
    
    Args:
        dataset: FITSDataset, IterableFITSDataset, or list of file paths
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data (ignored for IterableDataset)
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        prefetch_factor: Number of samples loaded in advance by each worker
        persistent_workers: Whether to keep workers alive between epochs
        **kwargs: Additional arguments passed to DataLoader
    
    Returns:
        Configured DataLoader instance
    """
    
    # Convert list of paths to FITSDataset if needed
    if isinstance(dataset, list):
        dataset = FITSDataset(dataset)
    
    # Adjust parameters based on dataset type
    if isinstance(dataset, IterableFITSDataset):
        # IterableDataset doesn't support shuffle
        shuffle = False
    
    # Set optimal defaults for astronomical data
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory and torch.cuda.is_available(),
        'prefetch_factor': prefetch_factor if num_workers > 0 else 2,
        'persistent_workers': persistent_workers and num_workers > 0,
        'drop_last': False,  # Keep all samples
    }
    
    # Update with user-provided kwargs
    dataloader_kwargs.update(kwargs)
    
    return DataLoader(dataset, **dataloader_kwargs)


def create_fits_dataloader(
    file_paths: List[str],
    hdu: int = 0,
    transform: Optional[Callable] = None,
    device: str = 'cpu',
    **dataloader_kwargs
) -> DataLoader:
    """
    Convenience function to create a DataLoader from FITS file paths.
    
    Args:
        file_paths: List of FITS file paths
        hdu: HDU index to read from each file
        transform: Optional transform to apply to each sample
        device: PyTorch device to load data onto
        **dataloader_kwargs: Arguments passed to create_dataloader
    
    Returns:
        Configured DataLoader instance
    """
    dataset = FITSDataset(
        file_paths=file_paths,
        hdu=hdu,
        transform=transform,
        device=device
    )
    
    return create_dataloader(dataset, **dataloader_kwargs)


def create_streaming_dataloader(
    index_url: str,
    hdu: int = 0,
    transform: Optional[Callable] = None,
    device: str = 'cpu',
    shard_size: int = 1000,
    **dataloader_kwargs
) -> DataLoader:
    """
    Create a streaming DataLoader for large-scale FITS datasets.
    
    Args:
        index_url: URL to sharded index file
        hdu: HDU index to read from each file
        transform: Optional transform to apply to each sample
        device: PyTorch device to load data onto
        shard_size: Number of files per shard
        **dataloader_kwargs: Arguments passed to create_dataloader
    
    Returns:
        Configured DataLoader for streaming
    """
    dataset = IterableFITSDataset(
        index_url=index_url,
        hdu=hdu,
        transform=transform,
        device=device,
        shard_size=shard_size
    )
    
    # Set defaults optimized for streaming
    streaming_defaults = {
        'batch_size': 32,
        'num_workers': 4,  # More workers for streaming
        'prefetch_factor': 4,  # Higher prefetch for streaming
        'persistent_workers': True,  # Keep workers alive
    }
    
    # Merge with user kwargs efficiently
    dataloader_kwargs = {**streaming_defaults, **dataloader_kwargs}
    
    return create_dataloader(dataset, **dataloader_kwargs)