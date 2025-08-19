"""
DataLoader factory functions for torchfits.

This module provides high-level functions to create optimized DataLoaders
for FITS datasets with appropriate defaults for astronomical data.
"""

from typing import Optional, Callable, Union, List, Dict, Any
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler

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


def create_distributed_dataloader(
    dataset: Union[FITSDataset, List[str]],
    batch_size: int = 32,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 0,
    **dataloader_kwargs
) -> DataLoader:
    """
    Create a distributed DataLoader for multi-GPU/multi-node training.
    
    Args:
        dataset: FITSDataset or list of file paths
        batch_size: Number of samples per batch per replica
        num_replicas: Number of processes participating in distributed training
        rank: Rank of the current process
        shuffle: Whether to shuffle the data
        seed: Random seed for shuffling
        **dataloader_kwargs: Additional arguments passed to create_dataloader
    
    Returns:
        Configured DataLoader with DistributedSampler
    """
    # Convert list of paths to FITSDataset if needed
    if isinstance(dataset, list):
        dataset = FITSDataset(dataset)
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=seed
    )
    
    # Remove shuffle from dataloader kwargs since sampler handles it
    dataloader_kwargs.pop('shuffle', None)
    
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # Handled by sampler
        **dataloader_kwargs
    )


def create_ml_dataloader(
    file_paths: List[str],
    batch_size: int = 32,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    random_seed: int = 42,
    transforms: Optional[Dict[str, Callable]] = None,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    Create train/validation/test DataLoaders for ML workflows.
    
    Args:
        file_paths: List of FITS file paths
        batch_size: Batch size for all splits
        validation_split: Fraction of data for validation
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducible splits
        transforms: Dict of transforms for each split ('train', 'val', 'test')
        **kwargs: Additional arguments for DataLoader creation
    
    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    import random
    
    # Shuffle file paths with fixed seed for reproducibility
    random.seed(random_seed)
    shuffled_paths = file_paths.copy()
    random.shuffle(shuffled_paths)
    
    # Calculate split indices
    total_files = len(shuffled_paths)
    test_size = int(total_files * test_split)
    val_size = int(total_files * validation_split)
    train_size = total_files - test_size - val_size
    
    # Split the data
    train_paths = shuffled_paths[:train_size]
    val_paths = shuffled_paths[train_size:train_size + val_size]
    test_paths = shuffled_paths[train_size + val_size:]
    
    # Default transforms
    if transforms is None:
        transforms = {}
    
    # Create datasets and dataloaders
    dataloaders = {}
    
    if train_paths:
        dataloaders['train'] = create_fits_dataloader(
            train_paths,
            batch_size=batch_size,
            shuffle=True,
            transform=transforms.get('train'),
            **kwargs
        )
    
    if val_paths:
        dataloaders['val'] = create_fits_dataloader(
            val_paths,
            batch_size=batch_size,
            shuffle=False,
            transform=transforms.get('val'),
            **kwargs
        )
    
    if test_paths:
        dataloaders['test'] = create_fits_dataloader(
            test_paths,
            batch_size=batch_size,
            shuffle=False,
            transform=transforms.get('test'),
            **kwargs
        )
    
    return dataloaders


def get_optimal_dataloader_config(
    dataset_size: int,
    file_size_mb: float,
    available_memory_gb: float,
    num_gpus: int = 1
) -> Dict[str, Any]:
    """
    Get optimal DataLoader configuration based on system resources.
    
    Args:
        dataset_size: Number of files in dataset
        file_size_mb: Average file size in MB
        available_memory_gb: Available system memory in GB
        num_gpus: Number of GPUs available
    
    Returns:
        Dictionary with optimal DataLoader parameters
    """
    # Calculate optimal batch size based on memory
    memory_per_sample_mb = file_size_mb * 2  # Account for processing overhead
    max_batch_size = int((available_memory_gb * 1024) / (memory_per_sample_mb * 4))  # Conservative estimate
    
    # Optimal number of workers
    import os
    cpu_count = os.cpu_count() or 4
    optimal_workers = min(cpu_count, 8)  # Cap at 8 workers
    
    # Adjust for dataset size
    if dataset_size < 1000:
        optimal_workers = min(optimal_workers, 2)
        batch_size = min(max_batch_size, 16)
    elif dataset_size < 10000:
        batch_size = min(max_batch_size, 32)
    else:
        batch_size = min(max_batch_size, 64)
    
    # GPU-specific adjustments
    if num_gpus > 1:
        batch_size = batch_size * num_gpus
        optimal_workers = optimal_workers * num_gpus
    
    return {
        'batch_size': max(1, batch_size),
        'num_workers': optimal_workers,
        'pin_memory': num_gpus > 0,
        'prefetch_factor': 4 if optimal_workers > 0 else 2,
        'persistent_workers': optimal_workers > 0 and dataset_size > 1000,
    }


# Export convenience function
def create_fits_ml_pipeline(
    file_paths: List[str],
    target_column: Optional[str] = None,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    Create a complete ML pipeline with train/val/test splits.
    
    This is a high-level convenience function that sets up everything
    needed for a typical ML workflow with FITS data.
    
    Args:
        file_paths: List of FITS file paths
        target_column: Name of target column for supervised learning
        **kwargs: Arguments passed to create_ml_dataloader
    
    Returns:
        Dictionary with configured DataLoaders
    """
    return create_ml_dataloader(file_paths, **kwargs)