"""
Logging utilities for torchfits.

Provides consistent logging and error handling across the library.
"""

import logging
import sys
from functools import wraps
from typing import Optional

# Configure torchfits logger
logger = logging.getLogger("torchfits")
logger.setLevel(logging.INFO)

# Create console handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_errors(func):
    """Decorator to log exceptions before re-raising."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise

    return wrapper


def log_performance(func):
    """Decorator to log performance metrics."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        import time

        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            logger.debug(
                f"{func.__name__} completed in {(end_time - start_time) * 1000:.2f}ms"
            )
            return result
        except Exception as e:
            end_time = time.perf_counter()
            logger.error(
                f"{func.__name__} failed after {(end_time - start_time) * 1000:.2f}ms: {str(e)}"
            )
            raise

    return wrapper


def set_log_level(level: str):
    """Set logging level for torchfits."""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logger.setLevel(level_map.get(level.upper(), logging.INFO))


def log_fits_error(operation: str, status: int, details: Optional[str] = None):
    """Log CFITSIO errors with context."""
    msg = f"CFITSIO error in {operation}: status={status}"
    if details:
        msg += f", details={details}"
    logger.error(msg)


def log_memory_usage(context: str, size_mb: float):
    """Log memory usage information."""
    logger.debug(f"Memory usage in {context}: {size_mb:.2f} MB")


def log_performance_warning(operation: str, time_ms: float, threshold_ms: float = 1000):
    """Log performance warnings for slow operations."""
    if time_ms > threshold_ms:
        logger.warning(
            f"Slow operation detected: {operation} took {time_ms:.2f}ms (threshold: {threshold_ms}ms)"
        )
