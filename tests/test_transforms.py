"""
Tests for torchfits transforms module.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from torchfits.transforms import (
    ZScale, AsinhStretch, LogStretch, PowerStretch, Normalize, MinMaxScale,
    RandomCrop, CenterCrop, RandomFlip, GaussianNoise, ToDevice, Compose,
    create_training_transform, create_validation_transform, create_inference_transform
)


class TestBasicTransforms:
    """Test basic transformation functions."""
    
    def test_zscale_normalization(self):
        """Test ZScale normalization."""
        # Create test data with known statistics
        data = torch.randn(100, 100) * 10 + 50
        transform = ZScale()
        
        result = transform(data)
        
        # Result should be roughly normalized
        assert result.min() >= 0
        assert result.max() <= 1
        assert result.shape == data.shape
    
    def test_asinh_stretch(self):
        """Test asinh stretch transformation."""
        data = torch.randn(50, 50)
        transform = AsinhStretch(a=0.1, Q=8.0)
        
        result = transform(data)
        
        assert result.shape == data.shape
        assert torch.isfinite(result).all()
    
    def test_log_stretch(self):
        """Test logarithmic stretch."""
        data = torch.abs(torch.randn(50, 50)) + 1  # Ensure positive values
        transform = LogStretch(a=1000.0)
        
        result = transform(data)
        
        assert result.shape == data.shape
        assert torch.isfinite(result).all()
        assert (result >= 0).all()
    
    def test_power_stretch(self):
        """Test power law stretch."""
        data = torch.abs(torch.randn(50, 50))  # Ensure non-negative
        transform = PowerStretch(gamma=0.5)
        
        result = transform(data)
        
        assert result.shape == data.shape
        assert torch.isfinite(result).all()
    
    def test_normalize(self):
        """Test standard normalization."""
        data = torch.randn(100, 100) * 5 + 10
        transform = Normalize()
        
        result = transform(data)
        
        assert result.shape == data.shape
        assert abs(result.mean().item()) < 0.1  # Should be close to 0
        assert abs(result.std().item() - 1.0) < 0.1  # Should be close to 1

    def test_minmax_scale(self):
        """Test min-max scaling."""
        data = torch.tensor([10.0, 20.0, 30.0])
        transform = MinMaxScale(min_val=0.0, max_val=1.0)
        
        result = transform(data)
        
        assert torch.allclose(result, torch.tensor([0.0, 0.5, 1.0]))



class TestGeometricTransforms:
    """Test geometric transformation functions."""
    
    def test_random_crop(self):
        """Test random crop transformation."""
        data = torch.randn(100, 100)
        transform = RandomCrop(50)
        
        result = transform(data)
        
        assert result.shape == (50, 50)
    
    def test_center_crop(self):
        """Test center crop transformation."""
        data = torch.randn(100, 100)
        transform = CenterCrop((60, 80))
        
        result = transform(data)
        
        assert result.shape == (60, 80)
    
    def test_random_flip(self):
        """Test random flip transformation."""
        data = torch.randn(50, 50)
        transform = RandomFlip(p=1.0)  # Always flip
        
        # Test multiple times to ensure randomness works
        results = [transform(data) for _ in range(5)]
        
        # At least some should be different from original
        assert any(not torch.equal(result, data) for result in results)
    
    def test_gaussian_noise(self):
        """Test Gaussian noise addition."""
        data = torch.zeros(50, 50)
        transform = GaussianNoise(std=0.1)
        
        result = transform(data)
        
        assert result.shape == data.shape
        assert result.std() > 0  # Should have added noise
    
    def test_to_device(self):
        """Test device transfer."""
        data = torch.randn(50, 50)
        transform = ToDevice('cpu')
        
        result = transform(data)
        
        assert result.device.type == 'cpu'
        assert torch.equal(result, data)


class TestCompose:
    """Test transform composition."""
    
    def test_compose_transforms(self):
        """Test composing multiple transforms."""
        data = torch.randn(100, 100)
        
        transforms = Compose([
            CenterCrop(80),
            ZScale(),
            GaussianNoise(std=0.01)
        ])
        
        result = transforms(data)
        
        assert result.shape == (80, 80)
        assert result.min() >= -0.1  # Roughly normalized with small noise
        assert result.max() <= 1.1
    
    def test_empty_compose(self):
        """Test empty composition."""
        data = torch.randn(50, 50)
        transforms = Compose([])
        
        result = transforms(data)
        
        assert torch.equal(result, data)


class TestConvenienceFunctions:
    """Test convenience transform creation functions."""
    
    def test_create_training_transform(self):
        """Test training transform creation."""
        transform = create_training_transform(crop_size=64, normalize=True, augment=True)
        
        data = torch.randn(100, 100)
        result = transform(data)
        
        assert result.shape == (64, 64)
    
    def test_create_validation_transform(self):
        """Test validation transform creation."""
        transform = create_validation_transform(crop_size=64, normalize=True)
        
        data = torch.randn(100, 100)
        result = transform(data)
        
        assert result.shape == (64, 64)
    
    def test_create_inference_transform(self):
        """Test inference transform creation."""
        transform = create_inference_transform(normalize=True)
        
        data = torch.randn(100, 100)
        result = transform(data)
        
        assert result.shape == data.shape


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_crop_larger_than_input(self):
        """Test cropping larger than input size."""
        data = torch.randn(50, 50)
        transform = RandomCrop(100)
        
        with pytest.raises(ValueError):
            transform(data)
    
    def test_zero_std_normalization(self):
        """Test normalization with zero standard deviation."""
        data = torch.ones(50, 50) * 5  # Constant tensor
        transform = Normalize()
        
        result = transform(data)
        
        # Should handle zero std gracefully
        assert result.shape == data.shape
        assert torch.isfinite(result).all()
    
    def test_negative_values_log_stretch(self):
        """Test log stretch with negative values."""
        data = torch.randn(50, 50)  # Can have negative values
        transform = LogStretch(a=1000.0)
        
        # Should handle negative values (though result may not be meaningful)
        result = transform(data)
        assert result.shape == data.shape


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))])
def test_transforms_on_device(device):
    """Test transforms work on different devices."""
    data = torch.randn(50, 50, device=device)
    
    transforms = [
        ZScale(),
        AsinhStretch(),
        Normalize(),
        RandomCrop(40),
        GaussianNoise(std=0.01)
    ]
    
    for transform in transforms:
        result = transform(data)
        assert result.device == data.device
        assert result.shape[-2:] == (40, 40) if isinstance(transform, RandomCrop) else result.shape == data.shape


def test_transform_reproducibility():
    """Test that transforms with fixed seeds are reproducible."""
    data = torch.randn(100, 100)
    
    # Set seed for reproducible random operations
    torch.manual_seed(42)
    transform1 = RandomCrop(50)
    result1 = transform1(data)
    
    torch.manual_seed(42)
    transform2 = RandomCrop(50)
    result2 = transform2(data)
    
    assert torch.equal(result1, result2)