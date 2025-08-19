"""
Master benchmark runner for torchfits.

Runs core benchmark suites and generates comprehensive performance report.
Includes both essential core benchmarks and optional exhaustive testing.
"""


import sys
import time
import subprocess
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def run_benchmark_script(script_name, description):
    """Run a benchmark script safely."""
    try:
        script_path = Path(__file__).parent / script_name
        if not script_path.exists():
            print(f"⚠️  Skipping {description}: {script_name} not found")
            return False
            
        print(f"\\n{'='*80}")
        print(f"RUNNING: {description}")
        print(f"Script: {script_name}")
        print('='*80)
        
        start_time = time.time()
        result = subprocess.run([sys.executable, str(script_path)], 
                              cwd=Path(__file__).parent)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully in {elapsed:.2f}s")
            return True
        else:
            print(f"❌ {description} failed after {elapsed:.2f}s")
            return False
            
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def print_system_info():
    """Print system information for benchmark context."""
    import platform
    import torch
    
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Check optional dependencies
    try:
        import astropy
        print(f"astropy: {astropy.__version__}")
    except ImportError:
        print("astropy: Not available")
    
    try:
        import fitsio
        print("fitsio: Available")
    except ImportError:
        print("fitsio: Not available")
    
    try:
        import matplotlib
        print(f"matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("matplotlib: Not available")
    
    print()

def main():
    """Run all benchmark suites."""
    parser = argparse.ArgumentParser(description='Run torchfits benchmark suite')
    parser.add_argument('--exhaustive', '-e', action='store_true',
                        help='Run exhaustive benchmarks (default: False)')
    parser.add_argument('--no-exhaustive', action='store_true',
                        help='Skip exhaustive benchmarks even if dependencies are available')
    args = parser.parse_args()
    
    print("🚀 TORCHFITS BENCHMARK SUITE")
    print("="*80)
    print()
    
    print_system_info()
    
    # Core benchmarks (essential for development)
    core_benchmarks = [
        ("benchmark_basic.py", "Basic Smoke Tests"),
        ("benchmark_core.py", "Core Features"),
        ("benchmark_ml.py", "ML Workflows"),
        ("benchmark_table.py", "Table Operations"),
    ]
    
    total_start = time.time()
    successes = 0
    failures = 0
    
    # Run core benchmarks
    print("🔧 CORE BENCHMARKS")
    print("These are essential benchmarks for development validation.")
    print()
    
    for script, description in core_benchmarks:
        if run_benchmark_script(script, description):
            successes += 1
        else:
            failures += 1
    
    # Determine if we should run comprehensive benchmarks
    print(f"\\n{'='*80}")
    print("🔬 COMPREHENSIVE BENCHMARKS")
    print("The exhaustive benchmark suite tests all data types, formats, and scenarios.")
    print("This includes:")
    print("- All data types (int8, int16, int32, float32, float64)")
    print("- All dimensions (1D spectra, 2D images, 3D cubes)")
    print("- MEF files, multi-MEFs, compressed files")
    print("- WCS-enabled files, scaled data, table files")
    print("- Performance comparison with astropy and fitsio")
    print("- Memory analysis and detailed reporting")
    print("- Comprehensive plots and CSV output")
    print()
    print("This can take significantly longer and requires optional dependencies.")
    print("(astropy, fitsio, matplotlib, seaborn, pandas, psutil)")
    print()
    
    # Check if exhaustive benchmarks should run
    if args.no_exhaustive:
        run_comprehensive = False
        print("Skipping comprehensive benchmarks (--no-exhaustive flag)")
    elif args.exhaustive:
        run_comprehensive = True
        print("Running comprehensive benchmarks (--exhaustive flag)")
    else:
        # Auto-detect: run if dependencies are available
        try:
            import astropy, fitsio, matplotlib, seaborn, pandas, psutil  # noqa: F401
            run_comprehensive = True
            print("Auto-detected all dependencies - running comprehensive benchmarks")
            print("(Use --no-exhaustive to skip, or --exhaustive to force)")
        except ImportError as e:
            run_comprehensive = False
            print(f"Skipping comprehensive benchmarks - missing dependency: {e}")
            print("(Use --exhaustive to attempt anyway)")
    
    if run_comprehensive:
        print("\\nRunning comprehensive benchmark suite...")
        if run_benchmark_script("benchmark_all.py", "Comprehensive Benchmark Suite"):
            successes += 1
            print("\\n📊 Comprehensive results should now be available in:")
            print("   - benchmark_results/exhaustive_results.csv")
            print("   - benchmark_results/exhaustive_summary.md") 
            print("   - benchmark_results/*.png (plots)")
        else:
            failures += 1
    
    # Final summary
    total_time = time.time() - total_start
    total_tests = successes + failures
    
    print(f"\\n{'='*80}")
    print("BENCHMARK SUITE COMPLETE")
    print("="*80)
    print(f"Total time: {total_time:.2f}s")
    print(f"Successes: {successes}/{total_tests}")
    print(f"Failures: {failures}/{total_tests}")
    
    if failures == 0:
        print("🎉 All benchmarks passed!")
        if run_comprehensive:
            print("\\n📈 Check the benchmark_results/ directory for comprehensive analysis.")
        return 0
    else:
        print(f"⚠️  {failures} benchmark(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
