#!/usr/bin/env python3
"""
TorchFits Benchmark Runner

Consolidated script to run all benchmark tests with proper reporting.
This replaces the need for separate benchmark runner scripts.

Usage:
    python -m tests.benchmark_runner              # Run all benchmarks  
    python -m tests.benchmark_runner --fast       # Run fast benchmarks only
    python -m tests.benchmark_runner --existential # Run existential justification tests
    python -m tests.benchmark_runner --pytorch-frame # Run PyTorch Frame integration tests
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_pytest_with_args(test_args, description):
    """Run pytest with specific arguments and handle results."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    cmd = ["python", "-m", "pytest"] + test_args + ["-v", "--tb=short"]
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            
        return result.returncode
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="TorchFits Benchmark Runner")
    parser.add_argument("--fast", action="store_true", 
                       help="Run only fast benchmark tests")
    parser.add_argument("--all", action="store_true",
                       help="Run all benchmark tests (default)")
    
    args = parser.parse_args()
    
    if not args.fast:
        args.all = True
    
    results = []
    test_file = "tests/test_official_benchmark_suite.py"
    
    if args.fast or args.all:
        # Quick performance tests
        result = run_pytest_with_args(
            [f"{test_file}::test_image_performance", 
             f"{test_file}::test_table_performance"],
            "Fast Performance Benchmarks"
        )
        results.append(result)
    
    if args.all:
        # Cutout performance tests
        result = run_pytest_with_args(
            [f"{test_file}::test_cutout_performance"],
            "Cutout Performance Tests"
        )
        results.append(result)
        
        # Column selection performance tests
        result = run_pytest_with_args(
            [f"{test_file}::test_column_selection_performance"],
            "Column Selection Performance Tests"
        )
        results.append(result)
        
        # Memory efficiency tests
        result = run_pytest_with_args(
            [f"{test_file}::test_memory_efficiency"],
            "Memory Efficiency Tests"
        )
        results.append(result)
        
        # Error handling and edge cases
        result = run_pytest_with_args(
            [f"{test_file}::test_error_handling"],
            "Error Handling Tests"
        )
        results.append(result)
        
        # Performance summary
        result = run_pytest_with_args(
            [f"{test_file}::test_performance_summary"],
            "Performance Summary Report"
        )
        results.append(result)
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r == 0)
    failed_tests = total_tests - passed_tests
    
    print(f"Total benchmark suites: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    
    if failed_tests == 0:
        print("🎉 All benchmarks passed!")
        return 0
    else:
        print(f"⚠️  {failed_tests} benchmark suite(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
