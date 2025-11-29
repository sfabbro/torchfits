"""
Cloud/HPC optimized caching for torchfits.

This module provides intelligent caching strategies for different environments
including local development, HPC clusters, and cloud platforms.
"""

import os
import psutil
from typing import Dict, Any, Optional
from pathlib import Path


class CacheConfig:
    """Cache configuration for different environments."""
    
    def __init__(self, max_files: int = 100, max_memory_mb: int = 1024, 
                 disk_cache_gb: int = 10, prefetch_enabled: bool = True):
        self.max_files = max_files
        self.max_memory_mb = max_memory_mb
        self.disk_cache_gb = disk_cache_gb
        self.prefetch_enabled = prefetch_enabled
    
    @classmethod
    def for_environment(cls) -> 'CacheConfig':
        """Auto-detect optimal cache configuration."""
        # Get system memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Detect environment
        if cls._is_hpc_environment():
            # HPC: Large memory, shared filesystem
            return cls(
                max_files=1000,
                max_memory_mb=int(memory_gb * 1024 * 0.3),  # 30% of memory
                disk_cache_gb=50,
                prefetch_enabled=True
            )
        elif cls._is_cloud_environment():
            # Cloud: Variable memory, network storage
            return cls(
                max_files=500,
                max_memory_mb=int(memory_gb * 1024 * 0.2),  # 20% of memory
                disk_cache_gb=20,
                prefetch_enabled=True
            )
        elif cls._is_gpu_environment():
            # GPU workstation: High memory, fast local storage
            return cls(
                max_files=200,
                max_memory_mb=int(memory_gb * 1024 * 0.4),  # 40% of memory
                disk_cache_gb=30,
                prefetch_enabled=True
            )
        else:
            # Default: Conservative settings
            return cls(
                max_files=100,
                max_memory_mb=min(2048, int(memory_gb * 1024 * 0.1)),
                disk_cache_gb=5,
                prefetch_enabled=False
            )
    
    @staticmethod
    def _is_hpc_environment() -> bool:
        """Detect HPC batch system environment."""
        hpc_vars = ['SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID', 'SGE_JOB_ID']
        return any(var in os.environ for var in hpc_vars)
    
    @staticmethod
    def _is_cloud_environment() -> bool:
        """Detect cloud platform environment."""
        cloud_vars = [
            'AWS_EXECUTION_ENV', 'AWS_LAMBDA_FUNCTION_NAME',
            'GOOGLE_CLOUD_PROJECT', 'GCLOUD_PROJECT',
            'AZURE_FUNCTIONS_ENVIRONMENT', 'WEBSITE_SITE_NAME',
            'KUBERNETES_SERVICE_HOST', 'K_SERVICE'
        ]
        return any(var in os.environ for var in cloud_vars)
    
    @staticmethod
    def _is_gpu_environment() -> bool:
        """Detect GPU environment."""
        try:
            import torch
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except ImportError:
            return False


class CacheManager:
    """Advanced cache manager with multiple strategies."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig.for_environment()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage_mb': 0,
            'disk_usage_gb': 0
        }
    
    def configure_cpp_cache(self):
        """Configure the C++ cache backend."""
        try:
            import torchfits.cpp as cpp
            if hasattr(cpp, 'configure_cache'):
                cpp.configure_cache(self.config.max_files, self.config.max_memory_mb)
        except (ImportError, AttributeError):
            # Fallback when C++ module not available or function missing
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            import torchfits.cpp as cpp
            cpp_size = cpp.get_cache_size() if hasattr(cpp, 'get_cache_size') else 0
        except (ImportError, AttributeError):
            cpp_size = 0
        
        return {
            **self._stats,
            'cpp_cache_size': cpp_size,
            'config': {
                'max_files': self.config.max_files,
                'max_memory_mb': self.config.max_memory_mb,
                'disk_cache_gb': self.config.disk_cache_gb,
                'prefetch_enabled': self.config.prefetch_enabled
            },
            'hit_rate': self._stats['hits'] / max(1, self._stats['hits'] + self._stats['misses'])
        }
    
    def clear(self):
        """Clear all caches."""
        try:
            import torchfits.cpp as cpp
            if hasattr(cpp, 'clear_file_cache'):
                cpp.clear_file_cache()
        except (ImportError, AttributeError):
            pass
        
        self._stats = {key: 0 for key in self._stats}
    
    def optimize_for_dataset(self, file_paths: list, avg_file_size_mb: float):
        """Optimize cache settings for a specific dataset."""
        total_size_gb = len(file_paths) * avg_file_size_mb / 1024
        
        # Adjust cache size based on dataset
        if total_size_gb < self.config.disk_cache_gb:
            # Dataset fits in cache - enable aggressive caching
            self.config.max_files = len(file_paths)
            self.config.prefetch_enabled = True
        else:
            # Large dataset - use LRU strategy
            optimal_files = int(self.config.disk_cache_gb * 1024 / avg_file_size_mb)
            self.config.max_files = min(optimal_files, 1000)
        
        self.configure_cpp_cache()


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
        _cache_manager.configure_cpp_cache()
    return _cache_manager


def configure_for_environment():
    """Auto-configure cache for different environments."""
    manager = get_cache_manager()
    manager.configure_cpp_cache()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return get_cache_manager().get_stats()


def clear_cache():
    """Clear all cached files."""
    get_cache_manager().clear()


def configure_cache(max_files: int, max_memory_mb: int, disk_cache_gb: int = 10):
    """Manually configure cache settings."""
    global _cache_manager
    config = CacheConfig(max_files, max_memory_mb, disk_cache_gb)
    _cache_manager = CacheManager(config)
    _cache_manager.configure_cpp_cache()


def optimize_for_dataset(file_paths: list, avg_file_size_mb: float = 10.0):
    """Optimize cache for a specific dataset."""
    get_cache_manager().optimize_for_dataset(file_paths, avg_file_size_mb)


def get_optimal_cache_config() -> Dict[str, Any]:
    """Get optimal cache configuration for current environment."""
    config = CacheConfig.for_environment()
    return {
        'max_files': config.max_files,
        'max_memory_mb': config.max_memory_mb,
        'disk_cache_gb': config.disk_cache_gb,
        'prefetch_enabled': config.prefetch_enabled,
        'environment': _detect_environment_type()
    }


def _detect_environment_type() -> str:
    """Detect the type of environment we're running in."""
    if CacheConfig._is_hpc_environment():
        return 'hpc'
    elif CacheConfig._is_cloud_environment():
        return 'cloud'
    elif CacheConfig._is_gpu_environment():
        return 'gpu_workstation'
    else:
        return 'local'