#!/usr/bin/env python3
"""
Comprehensive test runner for torchfits.
Runs all test suites and generates coverage reports.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def run_test_file(test_file, description):
    """Run a single test file."""
    try:
        print(f"\n{'='*60}")
        print(f"RUNNING: {description}")
        print(f"File: {test_file}")
        print('='*60)
        
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v"],
            cwd=Path(__file__).parent,
            capture_output=False
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} passed in {elapsed:.2f}s")
            return True
        else:
            print(f"❌ {description} failed after {elapsed:.2f}s")
            return False
            
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking Dependencies")
    print("-" * 30)
    
    required = ['torch', 'numpy', 'astropy']
    optional = ['psutil', 'matplotlib', 'pytest']
    
    missing_required = []
    missing_optional = []
    
    for dep in required:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} (required)")
            missing_required.append(dep)
    
    for dep in optional:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ⚠️  {dep} (optional)")
            missing_optional.append(dep)
    
    if missing_required:
        print(f"\n❌ Missing required dependencies: {', '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional dependencies: {', '.join(missing_optional)}")
        print("Some tests may be skipped.")
    
    print("\n✅ Dependency check passed")
    return True


def print_system_info():
    """Print system information."""
    import platform
    import torch
    
    print("💻 System Information")
    print("-" * 30)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    try:
        import astropy
        print(f"Astropy: {astropy.__version__}")
    except ImportError:
        print("Astropy: Not available")
    
    print()


def main():
    """Run all tests."""
    print("🧪 TorchFITS Test Suite")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        return 1
    
    print_system_info()
    
    # Define test files and descriptions
    test_files = [
        ("test_api.py", "Main API Tests"),
        ("test_core.py", "Core Functionality"),
        ("test_hdu.py", "HDU Classes"),
        ("test_transforms.py", "Transform Functions"),
        ("test_dataloader.py", "DataLoader Creation"),
        ("test_buffer.py", "Buffer Management"),
        ("test_wcs.py", "WCS Functionality"),
        ("test_cache.py", "Caching System"),
        ("test_table.py", "Table Reading"),
        ("test_compression.py", "Compression Support"),
        ("test_performance.py", "Performance Tests"),
        ("test_integration.py", "Integration Tests"),
    ]
    
    # Run tests
    total_start = time.time()
    passed = 0
    failed = 0
    
    print("🏃 Running Test Suite")
    print("-" * 30)
    
    for test_file, description in test_files:
        test_path = Path(__file__).parent / test_file
        
        if not test_path.exists():
            print(f"⚠️  Skipping {description}: {test_file} not found")
            continue
        
        if run_test_file(test_path, description):
            passed += 1
        else:
            failed += 1
    
    # Summary
    total_time = time.time() - total_start
    total_tests = passed + failed
    
    print(f"\n{'='*60}")
    print("TEST SUITE SUMMARY")
    print('='*60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Test files run: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        
        # Try to run coverage if available
        try:
            print("\n📊 Generating Coverage Report")
            subprocess.run([
                sys.executable, "-m", "pytest", 
                "--cov=torchfits", 
                "--cov-report=html",
                "--cov-report=term"
            ], cwd=Path(__file__).parent)
            print("Coverage report generated in htmlcov/")
        except Exception:
            print("Coverage reporting not available (install pytest-cov)")
        
        return 0
    else:
        print(f"\n❌ {failed} test file(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())