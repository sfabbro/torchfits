"""
Logging utilities for torchfits.

Provides consistent logging across the library.
"""

import logging
import sys

# Re-export stdlib levels so `torchfits.logging.DEBUG` works if the submodule
# shadows the standard `logging` module on the parent package.
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# Configure torchfits logger
logger = logging.getLogger("torchfits")
logger.setLevel(logging.INFO)

# Create console handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
