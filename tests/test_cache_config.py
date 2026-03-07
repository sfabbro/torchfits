from unittest.mock import MagicMock
from torchfits.cache import CacheConfig, get_optimal_cache_config
import torchfits.cache


def test_optimal_cache_config_no_psutil(monkeypatch):
    """Test fallback when psutil is not available."""
    monkeypatch.setattr(torchfits.cache, "psutil", None)

    # Ensure it doesn't think it's a GPU env
    # Using patch.dict or monkeypatch for torch is harder if it's not installed,
    # but CacheConfig._is_gpu_environment can be patched.
    monkeypatch.setattr(CacheConfig, "_is_gpu_environment", lambda: False)

    config = get_optimal_cache_config()

    assert config["max_files"] == 100
    assert config["max_memory_mb"] == 1024
    assert config["disk_cache_gb"] == 5
    assert config["prefetch_enabled"] is False
    assert config["environment"] == "local"


def test_optimal_cache_config_local(monkeypatch):
    """Test default local environment."""
    mock_psutil = MagicMock()
    # 16 GB in bytes
    mock_psutil.virtual_memory().total = 16 * (1024**3)
    monkeypatch.setattr(torchfits.cache, "psutil", mock_psutil)

    # Ensure no other env vars are present
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("AWS_EXECUTION_ENV", raising=False)

    # Mock GPU detection
    monkeypatch.setattr(CacheConfig, "_is_gpu_environment", lambda: False)

    config = get_optimal_cache_config()

    assert config["environment"] == "local"
    # 10% of 16GB = 1.6GB = 1638.4 MB -> int(1638.4) = 1638
    assert config["max_memory_mb"] == 1638
    assert config["disk_cache_gb"] == 5


def test_optimal_cache_config_hpc(monkeypatch):
    """Test HPC environment detection."""
    mock_psutil = MagicMock()
    # 64 GB
    mock_psutil.virtual_memory().total = 64 * (1024**3)
    monkeypatch.setattr(torchfits.cache, "psutil", mock_psutil)

    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    config = get_optimal_cache_config()

    assert config["environment"] == "hpc"
    # 30% of 64GB = 19.2GB = 19660.8 MB -> int() = 19660
    assert config["max_memory_mb"] == 19660
    assert config["max_files"] == 1000
    assert config["disk_cache_gb"] == 50
    assert config["prefetch_enabled"] is True


def test_optimal_cache_config_cloud(monkeypatch):
    """Test cloud environment detection."""
    mock_psutil = MagicMock()
    # 16 GB
    mock_psutil.virtual_memory().total = 16 * (1024**3)
    monkeypatch.setattr(torchfits.cache, "psutil", mock_psutil)

    monkeypatch.setenv("AWS_EXECUTION_ENV", "lambda")

    config = get_optimal_cache_config()

    assert config["environment"] == "cloud"
    # 20% of 16GB = 3.2GB = 3276.8 MB -> int() = 3276
    assert config["max_memory_mb"] == 3276
    assert config["max_files"] == 500
    assert config["disk_cache_gb"] == 20
    assert config["prefetch_enabled"] is True


def test_optimal_cache_config_gpu(monkeypatch):
    """Test GPU workstation detection."""
    mock_psutil = MagicMock()
    # 32 GB
    mock_psutil.virtual_memory().total = 32 * (1024**3)
    monkeypatch.setattr(torchfits.cache, "psutil", mock_psutil)

    # Mocking CacheConfig._is_gpu_environment directly
    monkeypatch.setattr(CacheConfig, "_is_gpu_environment", lambda: True)

    config = get_optimal_cache_config()

    assert config["environment"] == "gpu_workstation"
    # 40% of 32GB = 12.8GB = 13107.2 MB -> int() = 13107
    assert config["max_memory_mb"] == 13107
    assert config["max_files"] == 200
    assert config["disk_cache_gb"] == 30
    assert config["prefetch_enabled"] is True
