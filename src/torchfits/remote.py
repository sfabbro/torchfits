"""
Python wrapper for remote file functionality.

This provides access to the remote file capabilities that are implemented
in the C++ backend but not yet fully exposed to Python.
"""

import re
import os
from typing import Optional, Union, Dict, Any


class RemoteFetcher:
    """
    Python interface to remote file fetching capabilities.
    
    This mirrors the C++ RemoteFetcher functionality for URL detection
    and provides a consistent interface for remote file handling.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the remote fetcher.
        
        Parameters:
        -----------
        cache_dir : str, optional
            Directory for caching remote files. Defaults to system temp.
        """
        self.cache_dir = cache_dir or self._get_default_cache_dir()
    
    def is_remote(self, filename_or_url: Union[str, Dict[str, Any]]) -> bool:
        """
        Check if a filename/URL/fsspec dict refers to a remote resource.
        
        Parameters:
        -----------
        filename_or_url : str or dict
            The filename, URL, or fsspec dictionary to check
            
        Returns:
        --------
        bool
            True if the URL is remote, False otherwise
        """
        # Handle fsspec dictionaries
        if isinstance(filename_or_url, dict):
            return self._is_fsspec_remote(filename_or_url)
        
        # Handle string URLs
        url_lower = filename_or_url.lower()
        
        # Standard web protocols
        if (url_lower.startswith('http://') or 
            url_lower.startswith('https://') or 
            url_lower.startswith('ftp://') or
            url_lower.startswith('ftps://')):
            return True
            
        # Cloud storage protocols
        if (url_lower.startswith('s3://') or
            url_lower.startswith('gs://') or
            url_lower.startswith('azure://') or
            url_lower.startswith('az://')):
            return True
            
        # Network file systems
        if url_lower.startswith('nfs://'):
            return True
            
        return False
    
    def _is_fsspec_remote(self, fsspec_dict: Dict[str, Any]) -> bool:
        """
        Check if an fsspec dictionary refers to a remote resource.
        
        Parameters:
        -----------
        fsspec_dict : dict
            fsspec parameters dictionary
            
        Returns:
        --------
        bool
            True if the fsspec refers to a remote resource
        """
        protocol = fsspec_dict.get('protocol', '').lower()
        
        # Remote protocols
        remote_protocols = {
            'http', 'https', 'ftp', 'ftps', 
            's3', 'gcs', 'gs', 'azure', 'az',
            'hdfs', 'webhdfs', 'nfs'
        }
        
        return protocol in remote_protocols
    
    def fsspec_to_url(self, fsspec_dict: Dict[str, Any]) -> str:
        """
        Convert fsspec dictionary to URL.
        
        Parameters:
        -----------
        fsspec_dict : dict
            fsspec parameters dictionary
            
        Returns:
        --------
        str
            URL string
        """
        protocol = fsspec_dict.get('protocol', '')
        host = fsspec_dict.get('host', '')
        path = fsspec_dict.get('path', '')
        
        if protocol == 's3':
            bucket = fsspec_dict.get('bucket', '')
            return f"s3://{bucket}/{path.lstrip('/')}"
        elif protocol in ['gs', 'gcs']:
            bucket = fsspec_dict.get('bucket', '')
            return f"gs://{bucket}/{path.lstrip('/')}"
        elif protocol in ['http', 'https', 'ftp', 'ftps']:
            port = fsspec_dict.get('port', '')
            port_str = f":{port}" if port else ""
            return f"{protocol}://{host}{port_str}/{path.lstrip('/')}"
        else:
            # Generic URL construction
            return f"{protocol}://{host}/{path.lstrip('/')}"
    
    def _get_default_cache_dir(self) -> str:
        """Get the default cache directory."""
        # Match the C++ implementation
        cache_dir = os.environ.get('TORCHFITS_CACHE_DIR')
        if cache_dir:
            return cache_dir
        
        # Use system temp directory
        import tempfile
        return os.path.join(tempfile.gettempdir(), 'torchfits_cache')
    
    def get_cached_filename(self, filename_or_url: Union[str, Dict[str, Any]]) -> str:
        """
        Generate cached filename for a URL or fsspec dict.
        
        Parameters:
        -----------
        filename_or_url : str or dict
            The URL or fsspec dict to generate a cache filename for
            
        Returns:
        --------
        str
            Path to the cached file
        """
        # Convert fsspec dict to URL if needed
        if isinstance(filename_or_url, dict):
            url = self.fsspec_to_url(filename_or_url)
        else:
            url = filename_or_url
            
        # Simple hash-based naming (matches C++ logic)
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"remote_{url_hash}.fits")
    
    def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        return os.path.isfile(path)
    
    def ensure_local(self, filename_or_url: Union[str, Dict[str, Any]]) -> str:
        """
        Ensure a file is available locally.
        
        For remote URLs, this would download and cache the file.
        For local files, this returns the original path.
        
        Parameters:
        -----------
        filename_or_url : str or dict
            The file path, URL, or fsspec dictionary
            
        Returns:
        --------
        str
            Local path to the file
            
        Note:
        -----
        This Python implementation only provides URL detection.
        Actual downloading is handled by the C++ backend during read().
        """
        if self.is_remote(filename_or_url):
            # The C++ backend handles actual downloading during read()
            # This just returns what the cached path would be
            return self.get_cached_filename(filename_or_url)
        else:
            # For local files, return as-is (only works with string paths)
            if isinstance(filename_or_url, dict):
                raise ValueError("Local file access requires string path, not fsspec dict")
            return filename_or_url
