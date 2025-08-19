"""
Cloud/HPC optimized caching for torchfits.
"""

def configure_for_environment():
    """Auto-configure cache for different environments."""
    import os
    from . import cpp
    
    # Detect environment via environment variables only
    if 'SLURM_JOB_ID' in os.environ or 'PBS_JOBID' in os.environ or 'LSB_JOBID' in os.environ:
        # HPC batch system detected
        cpp.configure_cache(500, 2048)
    elif any(key in os.environ for key in ['AWS_EXECUTION_ENV', 'GOOGLE_CLOUD_PROJECT', 
                                          'AZURE_FUNCTIONS_ENVIRONMENT', 'KUBERNETES_SERVICE_HOST']):
        # Cloud environment detected
        cpp.configure_cache(200, 4096)
    else:
        # Default configuration
        cpp.configure_cache(100, 1024)

def get_cache_stats():
    """Get cache statistics."""
    from . import cpp
    return {
        'size': cpp.get_cache_size(),
        'configured': True
    }

def clear_cache():
    """Clear all cached files."""
    from . import cpp
    cpp.clear_file_cache()