"""
Data transformation modules for torchfits.

This module provides GPU-accelerated transformations for common
astronomical data preparation tasks (Phase 4 of implementation).
"""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor


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
        # Cache the log computation to avoid repeated calculation
        self._log_factor = torch.log(torch.tensor(self.a + 1))

    def __call__(self, tensor: Tensor) -> Tensor:
        """Apply log stretch."""
        return torch.log(self.a * tensor + 1) / self._log_factor

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


class MinMaxScale:
    """
    Min-max scaling to [min_val, max_val].
    """

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        """
        Initialize MinMaxScale.

        Args:
            min_val: Minimum value of output range
            max_val: Maximum value of output range
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor: Tensor) -> Tensor:
        """Apply min-max scaling."""
        t_min = tensor.min()
        t_max = tensor.max()

        # Prevent division by zero
        if torch.abs(t_max - t_min) < 1e-8:
            return torch.zeros_like(tensor)

        normalized = (tensor - t_min) / (t_max - t_min)
        return normalized * (self.max_val - self.min_val) + self.min_val

    def __repr__(self):
        return f"MinMaxScale(min_val={self.min_val}, max_val={self.max_val})"


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

        # Random crop position - use scalar tensor generation for efficiency
        i = torch.randint(0, h - th + 1, ()).item()
        j = torch.randint(0, w - tw + 1, ()).item()

        return tensor[..., i : i + th, j : j + tw]

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

        return tensor[..., i : i + th, j : j + tw]

    def __repr__(self):
        return f"CenterCrop(size={self.size})"


class RandomFlip:
    """Random flip transformation for data augmentation."""

    def __init__(self, horizontal: bool = True, vertical: bool = True, p: float = 0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p

    def __call__(self, tensor: Tensor) -> Tensor:
        if self.horizontal and torch.rand(1) < self.p:
            tensor = torch.flip(tensor, [-1])
        if self.vertical and torch.rand(1) < self.p:
            tensor = torch.flip(tensor, [-2])
        return tensor

    def __repr__(self):
        return f"RandomFlip(horizontal={self.horizontal}, vertical={self.vertical}, p={self.p})"


class GaussianNoise:
    """Add Gaussian noise for data augmentation."""

    def __init__(self, std: float = 0.01, snr_based: bool = False):
        self.std = std
        self.snr_based = snr_based

    def __call__(self, tensor: Tensor) -> Tensor:
        if self.snr_based:
            signal_std = tensor.std()
            noise_std = signal_std / self.std
        else:
            noise_std = self.std

        noise = torch.randn_like(tensor) * noise_std
        return tensor + noise

    def __repr__(self):
        return f"GaussianNoise(std={self.std}, snr_based={self.snr_based})"


class ToDevice:
    """Move tensor to specified device."""

    def __init__(self, device: Union[str, torch.device]):
        self.device = device

    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor.to(self.device)

    def __repr__(self):
        return f"ToDevice(device={self.device})"


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
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n    {0}".format(t)
        format_string += "\n)"
        return format_string


# Convenience functions
def create_display_transform(stretch: str = "asinh", **kwargs):
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
    if stretch == "asinh":
        transforms.append(AsinhStretch(**kwargs))
    elif stretch == "log":
        transforms.append(LogStretch(**kwargs))
    elif stretch == "power":
        transforms.append(PowerStretch(**kwargs))
    elif stretch == "zscale":
        transforms.append(ZScale(**kwargs))
    else:
        raise ValueError(f"Unknown stretch type: {stretch}")

    return Compose(transforms)


def create_training_transform(
    crop_size: int = 224, normalize: bool = True, augment: bool = True
):
    """
    Create a standard training transformation pipeline.

    Args:
        crop_size: Size for random crop
        normalize: Whether to apply normalization
        augment: Whether to apply data augmentation

    Returns:
        Composed transformation
    """
    transforms = [RandomCrop(crop_size)]

    if augment:
        transforms.extend([RandomFlip(p=0.5), GaussianNoise(std=0.01)])

    if normalize:
        transforms.append(ZScale())

    return Compose(transforms)


def create_validation_transform(crop_size: int = 224, normalize: bool = True):
    """
    Create a standard validation transformation pipeline.

    Args:
        crop_size: Size for center crop
        normalize: Whether to apply normalization

    Returns:
        Composed transformation
    """
    transforms = [CenterCrop(crop_size)]

    if normalize:
        transforms.append(ZScale())

    return Compose(transforms)


def create_inference_transform(normalize: bool = True):
    """
    Create a standard inference transformation pipeline.

    Args:
        normalize: Whether to apply normalization

    Returns:
        Composed transformation
    """
    transforms = []
    if normalize:
        transforms.append(ZScale())
    return Compose(transforms)


class PoissonNoise:
    """
    Add Poisson (shot) noise.

    Useful for simulating raw detector counts.
    """

    def __init__(self, lam: float = 1.0):
        """
        Args:
            lam: Scaling factor (gain). If input is flux, this converts to counts.
                 Output is converted back to original scale.
        """
        self.lam = lam

    def __call__(self, tensor: Tensor) -> Tensor:
        # Convert to counts (rate parameter)
        counts = torch.relu(tensor) * self.lam
        # Sample Poisson
        noisy_counts = torch.poisson(counts)
        # Convert back
        return noisy_counts / self.lam

    def __repr__(self):
        return f"PoissonNoise(lam={self.lam})"


class RandomRotation:
    """
    Random rotation for astronomical images (0-360 degrees).
    """

    def __init__(self, interpolation: str = "bilinear"):
        self.interpolation = interpolation

    def __call__(self, tensor: Tensor) -> Tensor:
        import math

        angle = torch.rand(1).item() * 360.0
        theta = math.radians(angle)

        # Create rotation matrix
        c, s = math.cos(theta), math.sin(theta)
        # PyTorch affine grid expects 2x3 matrix for 2D
        # [[cos, -sin, 0], [sin, cos, 0]]
        rot_mat = torch.tensor(
            [[c, -s, 0], [s, c, 0]], device=tensor.device, dtype=tensor.dtype
        )
        rot_mat = rot_mat.unsqueeze(0)  # Batch dim

        # Grid sample expects (N, C, H, W)
        if tensor.dim() == 3:
            x = tensor.unsqueeze(0)
        elif tensor.dim() == 2:
            x = tensor.unsqueeze(0).unsqueeze(0)
        else:
            x = tensor

        grid = torch.nn.functional.affine_grid(
            rot_mat.expand(x.size(0), -1, -1), x.size(), align_corners=False
        )
        x_rot = torch.nn.functional.grid_sample(
            x, grid, mode=self.interpolation, align_corners=False
        )

        if tensor.dim() == 3:
            return x_rot.squeeze(0)
        elif tensor.dim() == 2:
            return x_rot.squeeze(0).squeeze(0)
        return x_rot

    def __repr__(self):
        return f"RandomRotation(interpolation={self.interpolation})"


class RedshiftShift:
    """
    Simulate redshift effect on 1D spectra.

    Shifts the spectrum by (1+z).
    """

    def __init__(self, z_min: float = 0.0, z_max: float = 0.5):
        self.z_min = z_min
        self.z_max = z_max

    def __call__(self, tensor: Tensor) -> Tensor:
        # tensor shape: (C, L) or (L,)
        z = torch.rand(1).item() * (self.z_max - self.z_min) + self.z_min
        factor = 1.0 + z

        # We need to resample.
        # New grid corresponds to old grid * factor.
        # If we want to keep the same wavelength grid, we need to interpolate
        # from the shifted values back to the original grid.
        # f_obs(lambda) = f_rest(lambda / (1+z))

        orig_len = tensor.shape[-1]
        # Create grid [-1, 1]
        grid = torch.linspace(-1, 1, orig_len, device=tensor.device)

        # Shifted grid positions in original frame
        # We want to sample at x_new.
        # The value at x_new comes from x_old = x_new / (1+z) ?
        # No, lambda_obs = lambda_rest * (1+z)
        # So at lambda_obs, we see the flux from lambda_rest = lambda_obs / (1+z)

        # In grid coordinates [-1, 1]:
        # We effectively zoom in (if z > 0, features move to red/right).
        # So we sample from "left" (smaller indices).
        # grid_sample samples at (x, y).
        # We want to sample at grid / (1+z).
        # But grid is centered at 0. Wavelengths are usually positive.
        # This simple grid_sample assumes spatial [-1, 1].
        # For spectra, we usually want 1D interpolation on indices.

        # Let's use simple linear interpolation on indices
        indices = torch.arange(orig_len, device=tensor.device, dtype=tensor.dtype)
        # We want value at index i to come from index i / (1+z)
        # (Assuming linear wavelength spacing)
        # If log-linear, it's a constant shift.
        # Let's assume linear spacing for generic tensor.

        sample_indices = indices / factor

        # Clamp to valid range
        sample_indices = torch.clamp(sample_indices, 0, orig_len - 1)

        # Linear interpolation
        idx_floor = sample_indices.floor().long()
        idx_ceil = idx_floor + 1
        idx_ceil = torch.clamp(idx_ceil, 0, orig_len - 1)

        weight = sample_indices - idx_floor.float()

        if tensor.dim() == 1:
            val_floor = tensor[idx_floor]
            val_ceil = tensor[idx_ceil]
            return val_floor * (1 - weight) + val_ceil * weight
        else:
            # Handle (C, L)
            val_floor = tensor[:, idx_floor]
            val_ceil = tensor[:, idx_ceil]
            return val_floor * (1 - weight) + val_ceil * weight

    def __repr__(self):
        return f"RedshiftShift(z_min={self.z_min}, z_max={self.z_max})"


class PerturbByError:
    """
    Perturb data by its associated error.

    Useful for Monte Carlo error propagation or robust training.
    Expects input to be a tuple (data, error).
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, inputs: Tuple[Tensor, Tensor]) -> Tensor:
        data, error = inputs
        noise = torch.randn_like(data) * error * self.scale
        return data + noise

    def __repr__(self):
        return f"PerturbByError(scale={self.scale})"


class RobustScale:
    """
    Robust scaling using Median and Interquartile Range (IQR).

    (X - Median) / (Q75 - Q25)
    """

    def __init__(self, center: bool = True, scale: bool = True):
        self.center = center
        self.scale = scale

    def __call__(self, tensor: Tensor) -> Tensor:
        if self.center:
            median = torch.median(tensor)
            tensor = tensor - median

        if self.scale:
            q75 = torch.quantile(tensor, 0.75)
            q25 = torch.quantile(tensor, 0.25)
            iqr = q75 - q25
            if iqr > 1e-8:
                tensor = tensor / iqr

        return tensor

    def __repr__(self):
        return f"RobustScale(center={self.center}, scale={self.scale})"
