import torch

import torchfits.transforms as T


def test_poisson_noise():
    """Test PoissonNoise transform."""
    data = torch.ones(1000) * 100.0  # Mean 100
    noise = T.PoissonNoise(lam=1.0)
    noisy = noise(data)
    # Mean should be close to 100, variance close to 100
    assert torch.abs(noisy.mean() - 100.0) < 5.0
    assert torch.abs(noisy.var() - 100.0) < 20.0


def test_random_rotation():
    """Test RandomRotation transform."""
    # Create a 2D image with a gradient
    img = torch.zeros(10, 10)
    for i in range(10):
        img[i, :] = i
    rot = T.RandomRotation()
    rotated = rot(img)
    assert rotated.shape == img.shape


def test_redshift_shift():
    """Test RedshiftShift transform."""
    # Create a spectrum with a peak
    spec = torch.zeros(100)
    spec[50] = 1.0  # Peak at 50
    # Shift by z=0.1 -> factor 1.1
    # New peak should be at 50 * 1.1 = 55

    shifter = T.RedshiftShift(z_min=0.1, z_max=0.1)  # Fixed z=0.1
    shifted = shifter(spec)

    peak_idx = torch.argmax(shifted).item()
    assert abs(peak_idx - 55) <= 1  # Allow interpolation error


def test_perturb_by_error():
    """Test PerturbByError transform."""
    data = torch.zeros(1000)
    error = torch.ones(1000)
    perturb = T.PerturbByError(scale=1.0)
    perturbed = perturb((data, error))
    # Should be N(0, 1)
    assert torch.abs(perturbed.mean()) < 0.2
    assert torch.abs(perturbed.std() - 1.0) < 0.2


def test_robust_scale():
    """Test RobustScale transform."""
    data = torch.randn(1000) * 10 + 50
    # Add outliers
    data[0] = 1000.0
    scaler = T.RobustScale()
    scaled = scaler(data)
    # Median should be 0
    assert torch.abs(torch.median(scaled)) < 0.1
    # IQR should be 1
    q75 = torch.quantile(scaled, 0.75)
    q25 = torch.quantile(scaled, 0.25)
    assert torch.abs((q75 - q25) - 1.0) < 0.1
