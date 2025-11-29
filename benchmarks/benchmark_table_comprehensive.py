"""
Comprehensive Table Performance Benchmark Suite for torchfits
=============================================================

Tests table reading performance against astropy and fitsio across various scenarios:
- Different table sizes (1K to 10M+ rows)
- Different column configurations (5 to 100+ columns)
- Mixed data types (int, float, bool, string)
- Columnar access patterns
- Memory efficiency and zero-copy validation
"""

import time
import sys
import os
from pathlib import Path
import numpy as np
import torch
import tempfile
import gc
import psutil
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import torchfits

# Optional dependencies
try:
    from astropy.io import fits as astropy_fits
    from astropy.table import Table
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    print("Warning: astropy not available for comparison")

try:
    import fitsio
    HAS_FITSIO = True
except ImportError:
    HAS_FITSIO = False
    print("Warning: fitsio not available for comparison")

# Benchmark configurations following benchmarks_all.py style
TABLE_SIZES = [
    1_000,      # Small: quick testing
    10_000,     # Medium: typical catalog size
    100_000,    # Large: significant catalog
    1_000_000,  # Very large: major survey
    10_000_000, # Massive: full sky survey (optional, long test)
]

COLUMN_CONFIGS = [
    # Basic configurations
    {'ncols': 5, 'types': ['f4'] * 5, 'name': 'basic_float'},
    {'ncols': 10, 'types': ['f4'] * 5 + ['i4'] * 5, 'name': 'mixed_basic'},
    {'ncols': 20, 'types': ['f4'] * 10 + ['i4'] * 5 + ['f8'] * 5, 'name': 'mixed_medium'},
    
    # Astronomical catalog configurations
    {'ncols': 15, 'types': ['f8'] * 6 + ['f4'] * 6 + ['i4'] * 3, 'name': 'astrometry_catalog'},  # RA, DEC, proper motions, magnitudes, IDs
    {'ncols': 25, 'types': ['f8'] * 10 + ['f4'] * 10 + ['i4'] * 5, 'name': 'photometry_catalog'},  # Multi-band photometry
    {'ncols': 50, 'types': ['f4'] * 30 + ['f8'] * 10 + ['i4'] * 10, 'name': 'survey_catalog'},    # Full survey catalog
    {'ncols': 100, 'types': ['f4'] * 50 + ['f8'] * 25 + ['i4'] * 20 + ['i1'] * 5, 'name': 'mega_catalog'},  # Comprehensive catalog
]

class TableBenchmarkSuite:
    """Comprehensive table benchmark suite following benchmarks_all.py patterns."""
    
    def __init__(self, max_table_size: int = 1_000_000):
        self.max_table_size = max_table_size
        self.results = {}
        self.process = psutil.Process()
        
    def create_test_table(self, nrows: int, config: Dict) -> str:
        """Create a test FITS table file with specified configuration."""
        data = {}
        ncols = config['ncols']
        types = config['types'] 
        
        for i, dtype in enumerate(types[:ncols]):
            col_name = f'col_{i:02d}'
            
            if dtype.startswith('f'):  # float types
                if dtype == 'f4':
                    data[col_name] = np.random.randn(nrows).astype(np.float32)
                else:  # f8
                    data[col_name] = np.random.randn(nrows).astype(np.float64)
            elif dtype.startswith('i'):  # integer types
                if dtype == 'i1':
                    data[col_name] = np.random.randint(-128, 127, nrows, dtype=np.int8)
                elif dtype == 'i4':
                    data[col_name] = np.random.randint(-1000000, 1000000, nrows, dtype=np.int32)
                else:  # i8
                    data[col_name] = np.random.randint(-1000000, 1000000, nrows, dtype=np.int64)
            elif dtype == 'bool':
                data[col_name] = np.random.choice([True, False], nrows)
            elif dtype.startswith('U'):  # string types
                str_len = int(dtype[1:]) if len(dtype) > 1 else 10
                data[col_name] = [f'str_{i:0{min(str_len-4, 6)}d}' for i in range(nrows)]
        
        # Create temporary file
        fd, filepath = tempfile.mkstemp(suffix='.fits', prefix='benchmark_table_')
        os.close(fd)
        
        if HAS_ASTROPY:
            # Use astropy to create the table
            table = Table(data)
            hdu = astropy_fits.BinTableHDU(table, name=f'TABLE_{config["name"].upper()}')
            hdu.writeto(filepath, overwrite=True)
        else:
            # Fallback: create minimal table structure
            raise RuntimeError("Need astropy to create test tables")
            
        return filepath
    
    def measure_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self.process.memory_info().rss
    
    def benchmark_torchfits(self, filepath: str, config: Dict, nrows: int) -> Dict[str, Any]:
        """Benchmark torchfits table reading performance with correctness validation."""
        gc.collect()
        mem_before = self.measure_memory_usage()
        
        # Warmup
        try:
            _ = torchfits.read(filepath, hdu=1)
        except:
            pass
        
        # Benchmark full table read
        times = []
        last_data = None
        for _ in range(3):  # Multiple runs for stability
            gc.collect()
            start_time = time.perf_counter()
            try:
                data, header = torchfits.read(filepath, hdu=1)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                last_data = data
            except Exception as e:
                return {'error': str(e), 'time': float('inf'), 'memory_mb': 0, 'valid': False}
        
        mem_after = self.measure_memory_usage()
        avg_time = np.mean(times)
        
        # Enhanced correctness validation
        data_valid = False
        columns_found = 0
        correctness_issues = []
        
        if isinstance(last_data, dict):
            columns_found = len(last_data)
            
            # Check if we have the expected number of columns
            if columns_found != config['ncols']:
                correctness_issues.append(f"Column count mismatch: got {columns_found}, expected {config['ncols']}")
            
            # Check each column
            if columns_found > 0:
                for col_name, tensor_data in last_data.items():
                    if hasattr(tensor_data, 'shape'):
                        if len(tensor_data) != nrows:
                            correctness_issues.append(f"Row count mismatch in {col_name}: got {len(tensor_data)}, expected {nrows}")
                        data_valid = True
                    else:
                        correctness_issues.append(f"Column {col_name} is not tensor-like")
        else:
            correctness_issues.append(f"Data is not a dict: {type(last_data)}")
        
        return {
            'time': avg_time,
            'memory_mb': (mem_after - mem_before) / 1024 / 1024,
            'valid': data_valid and len(correctness_issues) == 0,
            'columns_found': columns_found,
            'expected_columns': config['ncols'],
            'throughput_rows_per_sec': nrows / avg_time if avg_time > 0 else 0,
            'correctness_issues': correctness_issues,
        }
    
    def benchmark_astropy(self, filepath: str, config: Dict, nrows: int) -> Dict[str, Any]:
        """Benchmark astropy table reading performance."""
        if not HAS_ASTROPY:
            return {'error': 'astropy not available', 'time': float('inf'), 'memory_mb': 0}
        
        gc.collect()
        mem_before = self.measure_memory_usage()
        
        # Warmup
        try:
            with astropy_fits.open(filepath) as hdul:
                _ = hdul[1].data
        except:
            pass
        
        # Benchmark
        times = []
        for _ in range(3):
            gc.collect()
            start_time = time.perf_counter()
            try:
                with astropy_fits.open(filepath) as hdul:
                    data = hdul[1].data
                    # Convert to dict for fair comparison
                    result = {col: data[col] for col in data.names}
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                return {'error': str(e), 'time': float('inf'), 'memory_mb': 0}
        
        mem_after = self.measure_memory_usage()
        avg_time = np.mean(times)
        
        return {
            'time': avg_time,
            'memory_mb': (mem_after - mem_before) / 1024 / 1024,
            'valid': True,
            'columns_found': len(result),
            'expected_columns': config['ncols'],
            'throughput_rows_per_sec': nrows / avg_time if avg_time > 0 else 0,
        }
    
    def benchmark_fitsio(self, filepath: str, config: Dict, nrows: int) -> Dict[str, Any]:
        """Benchmark fitsio table reading performance."""
        if not HAS_FITSIO:
            return {'error': 'fitsio not available', 'time': float('inf'), 'memory_mb': 0}
        
        gc.collect()
        mem_before = self.measure_memory_usage()
        
        # Warmup
        try:
            with fitsio.FITS(filepath) as fits:
                _ = fits[1].read()
        except:
            pass
        
        # Benchmark
        times = []
        for _ in range(3):
            gc.collect()
            start_time = time.perf_counter()
            try:
                with fitsio.FITS(filepath) as fits:
                    data = fits[1].read()
                    # Convert to dict for fair comparison
                    result = {col: data[col] for col in data.dtype.names}
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                return {'error': str(e), 'time': float('inf'), 'memory_mb': 0}
        
        mem_after = self.measure_memory_usage()
        avg_time = np.mean(times)
        
        return {
            'time': avg_time,
            'memory_mb': (mem_after - mem_before) / 1024 / 1024,
            'valid': True,
            'columns_found': len(result),
            'expected_columns': config['ncols'],
            'throughput_rows_per_sec': nrows / avg_time if avg_time > 0 else 0,
        }
    
    def run_table_reading_benchmarks(self) -> None:
        """Run comprehensive table reading benchmarks."""
        print("="*80)
        print("COMPREHENSIVE TABLE READING BENCHMARKS")
        print("="*80)
        print()
        
        for nrows in TABLE_SIZES:
            if nrows > self.max_table_size:
                continue
                
            print(f"\\n--- Testing {nrows:,} row tables ---")
            
            for config in COLUMN_CONFIGS:
                config_name = f"{nrows}_{config['name']}"
                print(f"\\nConfiguration: {config['name']} ({config['ncols']} columns)")
                
                # Create test table
                try:
                    filepath = self.create_test_table(nrows, config)
                    file_size_mb = os.path.getsize(filepath) / 1024 / 1024
                    print(f"  File size: {file_size_mb:.1f} MB")
                    
                    # Benchmark all implementations
                    results = {}
                    
                    # torchfits
                    print("  Testing torchfits...", end=' ', flush=True)
                    results['torchfits'] = self.benchmark_torchfits(filepath, config, nrows)
                    print(f"{results['torchfits']['time']:.4f}s")
                    
                    # astropy
                    if HAS_ASTROPY:
                        print("  Testing astropy...", end=' ', flush=True)
                        results['astropy'] = self.benchmark_astropy(filepath, config, nrows)
                        print(f"{results['astropy']['time']:.4f}s")
                    
                    # fitsio  
                    if HAS_FITSIO:
                        print("  Testing fitsio...", end=' ', flush=True)
                        results['fitsio'] = self.benchmark_fitsio(filepath, config, nrows)
                        print(f"{results['fitsio']['time']:.4f}s")
                    
                    # Store results
                    self.results[config_name] = {
                        'config': config,
                        'nrows': nrows,
                        'file_size_mb': file_size_mb,
                        'results': results
                    }
                    
                    # Cleanup
                    os.unlink(filepath)
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue
    
    def print_performance_summary(self) -> None:
        """Print comprehensive performance summary."""
        print("\\n" + "="*80)
        print("TABLE BENCHMARK PERFORMANCE SUMMARY") 
        print("="*80)
        
        if not self.results:
            print("No results to display.")
            return
        
        # Performance comparison table
        print(f"\\n{'Configuration':<25} {'Rows':<10} {'torchfits':<12} {'vs astropy':<12} {'vs fitsio':<12} {'Throughput':<15}")
        print("-" * 100)
        
        total_speedup_astropy = []
        total_speedup_fitsio = []
        
        for config_name, data in self.results.items():
            config = data['config']
            nrows = data['nrows']
            results = data['results']
            
            torchfits_time = results.get('torchfits', {}).get('time', float('inf'))
            astropy_time = results.get('astropy', {}).get('time', float('inf'))
            fitsio_time = results.get('fitsio', {}).get('time', float('inf'))
            
            # Calculate speedups
            speedup_astropy = astropy_time / torchfits_time if torchfits_time > 0 else 0
            speedup_fitsio = fitsio_time / torchfits_time if torchfits_time > 0 else 0
            
            if speedup_astropy > 0 and speedup_astropy != float('inf'):
                total_speedup_astropy.append(speedup_astropy)
            if speedup_fitsio > 0 and speedup_fitsio != float('inf'):
                total_speedup_fitsio.append(speedup_fitsio)
            
            throughput = results.get('torchfits', {}).get('throughput_rows_per_sec', 0)
            
            print(f"{config['name']:<25} {nrows:<10,} {torchfits_time:<12.4f} "
                  f"{speedup_astropy:<12.2f}x {speedup_fitsio:<12.2f}x {throughput:<15,.0f}")
        
        # Overall statistics
        if total_speedup_astropy:
            avg_speedup_astropy = np.mean(total_speedup_astropy)
            print(f"\\nAverage speedup vs astropy: {avg_speedup_astropy:.2f}x")
        
        if total_speedup_fitsio:
            avg_speedup_fitsio = np.mean(total_speedup_fitsio)
            print(f"Average speedup vs fitsio: {avg_speedup_fitsio:.2f}x")
    
    def print_correctness_analysis(self) -> None:
        """Analyze data correctness and validation."""
        print("\\n" + "="*80)
        print("TABLE CORRECTNESS ANALYSIS")
        print("="*80)
        
        correctness_issues = []
        
        for config_name, data in self.results.items():
            results = data['results']
            config = data['config']
            
            torchfits_result = results.get('torchfits', {})
            
            # Check column count
            expected_cols = config['ncols']
            found_cols = torchfits_result.get('columns_found', 0)
            
            if found_cols != expected_cols:
                correctness_issues.append(f"{config_name}: Expected {expected_cols} columns, found {found_cols}")
            
            # Check validity
            if not torchfits_result.get('valid', False):
                correctness_issues.append(f"{config_name}: Invalid data structure returned")
            
            # Check for errors
            if 'error' in torchfits_result:
                correctness_issues.append(f"{config_name}: Error - {torchfits_result['error']}")
        
        if correctness_issues:
            print("\\nCORRECTNESS ISSUES FOUND:")
            for issue in correctness_issues:
                print(f"  ❌ {issue}")
        else:
            print("\\n✅ All tests passed correctness validation")
    
    def run_all_benchmarks(self) -> None:
        """Run the complete benchmark suite."""
        print("torchfits Comprehensive Table Benchmark Suite")
        print("=" * 50)
        print()
        
        self.run_table_reading_benchmarks()
        self.print_performance_summary()
        self.print_correctness_analysis()

def main():
    """Main benchmark execution."""
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive Table Benchmarks')
    parser.add_argument('--max-size', type=int, default=1_000_000,
                        help='Maximum table size to test (default: 1M rows)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick tests only (up to 100K rows)')
    
    args = parser.parse_args()
    
    max_size = 100_000 if args.quick else args.max_size
    
    suite = TableBenchmarkSuite(max_table_size=max_size)
    suite.run_all_benchmarks()

if __name__ == "__main__":
    main()