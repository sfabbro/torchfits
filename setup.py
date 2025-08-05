# torchfits setuptools extension builder
from setuptools import setup
import os
import sys

# Add current directory to Python path so we can import build
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from build import create_extension, get_torch_extensions
    _, BuildExtension = get_torch_extensions()
    ext_module = create_extension()
    
    setup(
        ext_modules=[ext_module],
        cmdclass={'build_ext': BuildExtension}
    )
except ImportError as e:
    print(f"Warning: Could not build C++ extensions: {e}")
    print("Installing Python-only version...")
    setup()
except Exception as e:
    print(f"Warning: Extension building failed: {e}")
    print("Installing Python-only version...")
    setup()
