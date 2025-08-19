"""
Data transformation modules for torchfits.

This module provides GPU-accelerated transformations for common
astronomical data preparation tasks (Phase 4 of implementation).
"""

import torch
from torch import Tensor
from typing import Optional, Tuple, Union


class AsinhStretch:
    """
    Asinh stretch transformation for astronomical images.
    
    Commonly used for displaying images with high dynamic range.
    """
    
    def __init__(self, a: float = 0.1, Q: float = 8.0):
        """
        Initialize asinh stretch.
        
        Args:
            a: Softening parameter
            Q: Stretch parameter
        """
        self.a = a
        self.Q = Q
    
    def __call__(self, tensor: Tensor) -> Tensor:
        """Apply asinh stretch to tensor."""
        return torch.asinh(self.Q * tensor) / self.Q
    
    def __repr__(self):
        return f"AsinhStretch(a={self.a}, Q={self.Q})"


class ZScale:
    """
    ZScale normalization for astronomical images.
    
    Automatically determines optimal display range based on image statistics.
    """
    
    def __init__(self, contrast: float = 0.25, max_reject: float = 0.5):
        """
        Initialize ZScale.
        
        Args:
            contrast: Contrast parameter
            max_reject: Maximum fraction of pixels to reject
        """
        self.contrast = contrast
        self.max_reject = max_reject
    
    def __call__(self, tensor: Tensor) -> Tensor:
        """Apply ZScale normalization."""
        # Simplified implementation - full version would use iterative algorithm
        z1 = torch.quantile(tensor, 0.05)
        z2 = torch.quantile(tensor, 0.95)
        
        # Prevent division by zero
        range_val = z2 - z1
        if torch.abs(range_val) < 1e-8:
            return torch.zeros_like(tensor)
        
        # Normalize to [0, 1]
        normalized = (tensor - z1) / range_val
        return torch.clamp(normalized, 0, 1)
    
    def __repr__(self):
        return f"ZScale(contrast={self.contrast}, max_reject={self.max_reject})"


class LogStretch:
    """
    Logarithmic stretch transformation.
    """
    
    def __init__(self, a: float = 1000.0):
        """
        Initialize log stretch.
        
        Args:
            a: Scaling parameter
        """
        self.a = a
    
    def __call__(self, tensor: Tensor) -> Tensor:
        """Apply log stretch."""
        return torch.log(self.a * tensor + 1) / torch.log(torch.tensor(self.a + 1))
    
    def __repr__(self):
        return f"LogStretch(a={self.a})"


class PowerStretch:
    """
    Power law stretch transformation.
    """
    
    def __init__(self, gamma: float = 0.5):
        """
        Initialize power stretch.
        
        Args:
            gamma: Power law exponent
        """
        self.gamma = gamma
    
    def __call__(self, tensor: Tensor) -> Tensor:
        """Apply power stretch."""
        return torch.pow(tensor, self.gamma)
    
    def __repr__(self):
        return f"PowerStretch(gamma={self.gamma})"


class Normalize:
    """
    Standard normalization (mean=0, std=1).
    """
    
    def __init__(self, mean: Optional[float] = None, std: Optional[float] = None):
        """
        Initialize normalization.
        
        Args:
            mean: Mean value (computed from data if None)
            std: Standard deviation (computed from data if None)
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: Tensor) -> Tensor:
        """Apply normalization."""
        mean = self.mean if self.mean is not None else tensor.mean()
        std = self.std if self.std is not None else tensor.std()
        
        # Prevent division by zero
        if torch.abs(std) < 1e-8:
            return tensor - mean
        
        return (tensor - mean) / std
    
    def __repr__(self):
        return f"Normalize(mean={self.mean}, std={self.std})"


class RandomCrop:
    """
    Random crop transformation for data augmentation.
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        """
        Initialize random crop.
        
        Args:
            size: Output size (height, width) or single int for square
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, tensor: Tensor) -> Tensor:
        """Apply random crop."""
        h, w = tensor.shape[-2:]
        th, tw = self.size
        
        if h < th or w < tw:
            raise ValueError(f"Input size {(h, w)} smaller than crop size {self.size}")
        
        # Random crop position
        i = torch.randint(0, h - th + 1, (1,)).item()
        j = torch.randint(0, w - tw + 1, (1,)).item()
        
        return tensor[..., i:i+th, j:j+tw]
    
    def __repr__(self):
        return f"RandomCrop(size={self.size})"


class CenterCrop:
    """
    Center crop transformation.
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        """
        Initialize center crop.
        
        Args:
            size: Output size (height, width) or single int for square
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, tensor: Tensor) -> Tensor:
        """Apply center crop."""
        h, w = tensor.shape[-2:]
        th, tw = self.size
        
        if h < th or w < tw:
            raise ValueError(f"Input size {(h, w)} smaller than crop size {self.size}")
        
        # Center crop position
        i = (h - th) // 2
        j = (w - tw) // 2
        
        return tensor[..., i:i+th, j:j+tw]
    
    def __repr__(self):
        return f"CenterCrop(size={self.size})"


class Compose:
    """
    Compose multiple transformations.
    """
    
    def __init__(self, transforms):
        """
        Initialize composition.
        
        Args:
            transforms: List of transform functions
        """
        self.transforms = transforms
    
    def __call__(self, tensor: Tensor) -> Tensor:
        """Apply all transformations in sequence."""
        for transform in self.transforms:
            tensor = transform(tensor)
        return tensor
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    {0}'.format(t)
        format_string += '\n)'
        return format_string


# Convenience functions
def create_display_transform(stretch: str = 'asinh', **kwargs):
    """
    Create a standard display transformation pipeline.
    
    Args:
        stretch: Type of stretch ('asinh', 'log', 'power', 'zscale')
        **kwargs: Parameters for the stretch function
    
    Returns:
        Composed transformation
    """
    transforms = []
    
    # Add stretch transformation
    if stretch == 'asinh':
        transforms.append(AsinhStretch(**kwargs))
    elif stretch == 'log':
        transforms.append(LogStretch(**kwargs))
    elif stretch == 'power':
        transforms.append(PowerStretch(**kwargs))
    elif stretch == 'zscale':
        transforms.append(ZScale(**kwargs))
    else:
        raise ValueError(f"Unknown stretch type: {stretch}")
    
    return Compose(transforms)


def create_training_transform(crop_size: int = 224, normalize: bool = True):
    """
    Create a standard training transformation pipeline.
    
    Args:
        crop_size: Size for random crop
        normalize: Whether to apply normalization
    
    Returns:
        Composed transformation
    """
    transforms = [RandomCrop(crop_size)]
    
    if normalize:
        transforms.append(Normalize())
    
    return Compose(transforms)