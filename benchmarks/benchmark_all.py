#!/usr/bin/env python3
"""
Exhaustive torchfits benchmark suite.

Tests all data types, formats, and operations with detailed reporting.
Covers: spectra, images, cubes, tables, MEFs, multi-MEFs, cutouts, 
multi-cutouts, multi-files, compression, WCS, scaling, all sizes.

Produces comprehensive tables, plots, and summaries.
"""

import sys
import time
import tempfile
import gc
import tracemalloc
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from statistics import mean, stdev
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torchfits
from astropy.io import fits as astropy_fits
from astropy.io.fits import CompImageHDU
from astropy.wcs import WCS
import fitsio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import psutil


class ExhaustiveBenchmarkSuite:
    """
    Exhaustive benchmark suite for torchfits covering all use cases.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="torchfits_exhaustive_"))
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.csv_file = self.output_dir / "exhaustive_results.csv"
        self.summary_file = self.output_dir / "exhaustive_summary.md"
        
        # Test configurations
        self.data_types = {
            'int8': (np.int8, 'BYTE_IMG'),
            'int16': (np.int16, 'SHORT_IMG'), 
            'int32': (np.int32, 'LONG_IMG'),
            'float32': (np.float32, 'FLOAT_IMG'),
            'float64': (np.float64, 'DOUBLE_IMG')
        }
        
        self.size_categories = {
            'tiny': {'1d': 1000, '2d': (64, 64), '3d': (5, 32, 32)},
            'small': {'1d': 10000, '2d': (256, 256), '3d': (10, 128, 128)},
            'medium': {'1d': 100000, '2d': (1024, 1024), '3d': (25, 256, 256)},
            'large': {'1d': 1000000, '2d': (2048, 2048), '3d': (50, 512, 512)}
        }
        
        self.compression_types = ['RICE_1', 'GZIP_1', 'GZIP_2', 'HCOMPRESS_1']
        
    def create_test_files(self) -> Dict[str, Path]:
        """Create comprehensive test dataset covering all scenarios."""
        print("Creating exhaustive test dataset...")
        files = {}
        
        # 1. Single HDU files (all combinations)
        files.update(self._create_single_hdu_files())
        
        # 2. MEF files (Multiple Extension FITS)
        files.update(self._create_mef_files())
        
        # 3. Multi-MEF files (MEFs with many extensions)
        files.update(self._create_multi_mef_files())
        
        # 4. Table files
        files.update(self._create_table_files())
        
        # 5. Scaled data files (BSCALE/BZERO)
        files.update(self._create_scaled_files())
        
        # 6. WCS-enabled files
        files.update(self._create_wcs_files())
        
        # 7. Compressed files
        files.update(self._create_compressed_files())
        
        # 8. Multi-file collections
        files.update(self._create_multi_file_collections())
        
        print(f"✓ Created {len(files)} test files in {self.temp_dir}")
        return files
    
    def _create_single_hdu_files(self) -> Dict[str, Path]:
        """Create single HDU files for all data types and sizes."""
        files = {}
        
        for size_name, size_specs in self.size_categories.items():
            for dtype_name, (np_dtype, fits_bitpix) in self.data_types.items():
                for dim_name, shape in size_specs.items():
                    # Skip huge 3D arrays to avoid memory issues
                    if size_name == 'large' and dim_name == '3d':
                        continue
                        
                    data = self._generate_data(shape, np_dtype)
                    filename = self.temp_dir / f"{size_name}_{dtype_name}_{dim_name}.fits"
                    
                    astropy_fits.PrimaryHDU(data).writeto(filename, overwrite=True)
                    files[f"{size_name}_{dtype_name}_{dim_name}"] = filename
        
        return files
    
    def _create_mef_files(self) -> Dict[str, Path]:
        """Create Multi-Extension FITS files."""
        files = {}
        
        for size_name in ['small', 'medium']:
            # Standard MEF with mixed data types
            hdu_list = [astropy_fits.PrimaryHDU()]
            
            shape = self.size_categories[size_name]['2d']
            for i, (dtype_name, (np_dtype, _)) in enumerate(self.data_types.items()):
                if i >= 3:  # Limit to 3 extensions
                    break
                data = self._generate_data(shape, np_dtype)
                hdu = astropy_fits.ImageHDU(data, name=f'EXT_{dtype_name.upper()}')
                hdu_list.append(hdu)
            
            mef_filename = self.temp_dir / f"mef_{size_name}.fits"
            astropy_fits.HDUList(hdu_list).writeto(mef_filename, overwrite=True)
            files[f"mef_{size_name}"] = mef_filename
        
        return files
    
    def _create_multi_mef_files(self) -> Dict[str, Path]:
        """Create MEF files with many extensions."""
        files = {}
        
        
        # Create MEF with 10 extensions
        hdu_list = [astropy_fits.PrimaryHDU()]
        shape = (256, 256)
        
        for i in range(10):
            dtype_name, (np_dtype, _) = list(self.data_types.items())[i % len(self.data_types)]
            data = self._generate_data(shape, np_dtype)
            hdu = astropy_fits.ImageHDU(data, name=f'EXT_{i:02d}_{dtype_name.upper()}')
            hdu_list.append(hdu)
        
        multi_mef_filename = self.temp_dir / "multi_mef_10ext.fits"
        astropy_fits.HDUList(hdu_list).writeto(multi_mef_filename, overwrite=True)
        files["multi_mef_10ext"] = multi_mef_filename
        
        return files
    
    def _create_table_files(self) -> Dict[str, Path]:
        """Create table FITS files."""
        files = {}
        
        
        for size_name in ['small', 'medium', 'large']:
            nrows = {
                'small': 1000,
                'medium': 10000, 
                'large': 100000
            }[size_name]
            
            # Create table data
            cols = [
                astropy_fits.Column(name='ID', format='J', array=np.arange(nrows)),
                astropy_fits.Column(name='RA', format='D', array=np.random.uniform(0, 360, nrows)),
                astropy_fits.Column(name='DEC', format='D', array=np.random.uniform(-90, 90, nrows)),
                astropy_fits.Column(name='FLUX', format='E', array=np.random.lognormal(0, 1, nrows)),
                astropy_fits.Column(name='MAG', format='E', array=np.random.normal(20, 2, nrows)),
                astropy_fits.Column(name='CLASS', format='10A', array=[f'STAR_{i%3}' for i in range(nrows)])
            ]
            
            table_hdu = astropy_fits.BinTableHDU.from_columns(cols, name='CATALOG')
            hdul = astropy_fits.HDUList([astropy_fits.PrimaryHDU(), table_hdu])
            
            table_filename = self.temp_dir / f"table_{size_name}.fits"
            hdul.writeto(table_filename, overwrite=True)
            files[f"table_{size_name}"] = table_filename
        
        return files
    
    def _create_scaled_files(self) -> Dict[str, Path]:
        """Create files with BSCALE/BZERO scaling."""
        files = {}
        
        
        for size_name in ['small', 'medium']:
            shape = self.size_categories[size_name]['2d']
            
            # Create float data that will be scaled to int16
            float_data = np.random.randn(*shape).astype(np.float32) * 1000 + 32768
            
            hdu = astropy_fits.PrimaryHDU()
            hdu.data = float_data.astype(np.int16)
            hdu.header['BSCALE'] = 0.1
            hdu.header['BZERO'] = 32768
            hdu.header['COMMENT'] = 'Scaled data test'
            
            scaled_filename = self.temp_dir / f"scaled_{size_name}.fits"
            hdu.writeto(scaled_filename, overwrite=True)
            files[f"scaled_{size_name}"] = scaled_filename
        
        return files
    
    def _create_wcs_files(self) -> Dict[str, Path]:
        """Create files with WCS information."""
        files = {}
        
        
        shape = (512, 512)
        data = self._generate_data(shape, np.float32)
        
        # Create WCS header
        hdu = astropy_fits.PrimaryHDU(data)
        hdu.header['CRPIX1'] = shape[1] / 2
        hdu.header['CRPIX2'] = shape[0] / 2
        hdu.header['CRVAL1'] = 180.0
        hdu.header['CRVAL2'] = 0.0
        hdu.header['CDELT1'] = -0.0001
        hdu.header['CDELT2'] = 0.0001
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'
        hdu.header['CUNIT1'] = 'deg'
        hdu.header['CUNIT2'] = 'deg'
        
        wcs_filename = self.temp_dir / "wcs_image.fits"
        hdu.writeto(wcs_filename, overwrite=True)
        files["wcs_image"] = wcs_filename
        
        return files
    
    def _create_compressed_files(self) -> Dict[str, Path]:
        """Create compressed FITS files."""
        files = {}
        
        
        shape = (1024, 1024)
        data = self._generate_data(shape, np.float32)
        
        for comp_type in self.compression_types:
            try:
                comp_hdu = CompImageHDU(data, compression_type=comp_type)
                hdul = astropy_fits.HDUList([astropy_fits.PrimaryHDU(), comp_hdu])
                
                comp_filename = self.temp_dir / f"compressed_{comp_type.lower()}.fits"
                hdul.writeto(comp_filename, overwrite=True)
                files[f"compressed_{comp_type.lower()}"] = comp_filename
            except Exception as e:
                print(f"Warning: Could not create {comp_type} compressed file: {e}")
        
        return files
    
    def _create_multi_file_collections(self) -> Dict[str, Path]:
        """Create collections for multi-file operations."""
        files = {}
        
        
        # Create a set of related files (simulating a time series)
        collection_dir = self.temp_dir / "timeseries"
        collection_dir.mkdir(exist_ok=True)
        
        shape = (256, 256)
        for i in range(5):
            data = self._generate_data(shape, np.float32) + i * 100  # Add offset per file
            filename = collection_dir / f"frame_{i:03d}.fits"
            astropy_fits.PrimaryHDU(data).writeto(filename, overwrite=True)
            files[f"timeseries_frame_{i:03d}"] = filename
        
        return files
    
    def _generate_data(self, shape, dtype):
        """Generate test data with appropriate values for data type."""
        if dtype == np.int8:
            return np.random.randint(-100, 100, shape, dtype=dtype)
        elif dtype == np.int16:
            return np.random.randint(-1000, 1000, shape, dtype=dtype)
        elif dtype == np.int32:
            return np.random.randint(-10000, 10000, shape, dtype=dtype)
        else:
            if isinstance(shape, tuple):
                return np.random.randn(*shape).astype(dtype)
            else:
                return np.random.randn(shape).astype(dtype)
    
    def run_exhaustive_benchmarks(self, files: Dict[str, Path]) -> List[Dict]:
        """Run benchmarks on all test files."""
        print("\\n" + "=" * 100)
        print("EXHAUSTIVE BENCHMARK SUITE")
        print("=" * 100)
        
        csv_headers = [
            'filename', 'file_type', 'size_mb', 'data_type', 'dimensions', 'compression',
            'torchfits_mean', 'torchfits_std', 'torchfits_memory', 'torchfits_peak_memory',
            'astropy_mean', 'astropy_std', 'astropy_memory', 'astropy_peak_memory',
            'fitsio_mean', 'fitsio_std', 'fitsio_memory', 'fitsio_peak_memory',
            'astropy_torch_mean', 'astropy_torch_std', 'astropy_torch_memory', 'astropy_torch_peak_memory',
            'fitsio_torch_mean', 'fitsio_torch_std', 'fitsio_torch_memory', 'fitsio_torch_peak_memory',
            'best_method', 'torchfits_rank', 'speedup_vs_best'
        ]
        
        detailed_results = []
        
        for name, filepath in sorted(files.items()):
            result = self._benchmark_single_file(name, filepath)
            if result:
                detailed_results.append(result)
        
        # Write CSV results
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(detailed_results)
        
        print(f"\\n✓ Detailed results saved to: {self.csv_file}")
        return detailed_results
    
    def _benchmark_single_file(self, name: str, filepath: Path) -> Optional[Dict]:
        """Benchmark a single file with all methods."""
        size_mb = filepath.stat().st_size / 1024 / 1024
        
        # Parse file characteristics
        parts = name.split('_')
        file_type = self._get_file_type(name)
        data_type = next((p for p in parts if p in self.data_types.keys()), 'unknown')
        dimensions = next((p for p in parts if p in ['1d', '2d', '3d']), 'unknown')
        compression = self._get_compression_type(name)
        
        print(f"\\n{name} ({size_mb:.2f} MB) - {file_type} {data_type} {dimensions} {compression}")
        print("-" * 80)
        
        result = {
            'filename': name, 'file_type': file_type, 'size_mb': size_mb, 
            'data_type': data_type, 'dimensions': dimensions, 'compression': compression
        }
        
        # Determine HDU index for compressed files
        hdu_num = 1 if compression != 'uncompressed' else 0
        
        # Define benchmark methods
        methods = {}
        
        # Always test torchfits
        methods['torchfits'] = lambda: torchfits.read(str(filepath), hdu=hdu_num)
        
        # Test astropy if available
        methods['astropy'] = lambda: self._astropy_read(filepath, hdu_num)
        methods['astropy_torch'] = lambda: self._astropy_to_torch(filepath, hdu_num)
        
        # Test fitsio if available  
        methods['fitsio'] = lambda: fitsio.read(str(filepath), ext=hdu_num)
        methods['fitsio_torch'] = lambda: self._fitsio_to_torch(filepath, hdu_num)
        
        # Run benchmarks
        method_results = {}
        for method_name, method_func in methods.items():
            method_result = self._time_method(method_func, method_name, runs=3)
            method_results[method_name] = method_result
            
            if method_result:
                result[f'{method_name}_mean'] = method_result['mean']
                result[f'{method_name}_std'] = method_result['std']
                result[f'{method_name}_memory'] = method_result['memory']
                result[f'{method_name}_peak_memory'] = method_result['peak_memory']
                
                print(f"{method_name:15s}: {method_result['mean']:.6f}s ± {method_result['std']:.6f}s  "
                      f"mem: {method_result['memory']:.1f}MB  peak: {method_result['peak_memory']:.1f}MB")
            else:
                result[f'{method_name}_mean'] = None
                result[f'{method_name}_std'] = None
                result[f'{method_name}_memory'] = None
                result[f'{method_name}_peak_memory'] = None
                print(f"{method_name:15s}: FAILED")
        
        # Analyze results
        valid_methods = {k: v['mean'] for k, v in method_results.items() if v and v['mean'] is not None}
        if valid_methods:
            best_method = min(valid_methods.keys(), key=lambda k: valid_methods[k])
            sorted_methods = sorted(valid_methods.items(), key=lambda x: x[1])
            torchfits_rank = next((i+1 for i, (k, v) in enumerate(sorted_methods) if k == 'torchfits'), len(sorted_methods)+1)
            
            result['best_method'] = best_method
            result['torchfits_rank'] = torchfits_rank
            
            # Calculate speedup vs best
            if 'torchfits' in valid_methods:
                best_time = valid_methods[best_method]
                tf_time = valid_methods['torchfits']
                speedup = best_time / tf_time if best_method != 'torchfits' else tf_time / min(v for k, v in valid_methods.items() if k != 'torchfits')
                result['speedup_vs_best'] = speedup
                
                print(f"\\nBest method: {best_method} ({valid_methods[best_method]:.6f}s)")
                print(f"torchfits rank: {torchfits_rank}/{len(valid_methods)}")
                if best_method != 'torchfits':
                    print(f"torchfits vs best: {tf_time/valid_methods[best_method]:.2f}x")
            else:
                result['speedup_vs_best'] = None
        else:
            result['best_method'] = 'none'
            result['torchfits_rank'] = 999
            result['speedup_vs_best'] = None
        
        return result
    
    def _time_method(self, method_func, method_name: str, runs: int = 3) -> Optional[Dict]:
        """Time a method with memory monitoring."""
        times = []
        memory_usage = []
        peak_memory_usage = []
        
        for i in range(runs):
            try:
                gc.collect()
                for _ in range(3):  # Extra cleanup
                    gc.collect()
                
                # Start memory tracing
                tracemalloc.start()
                
                # Time the operation
                start_time = time.perf_counter()
                data = method_func()
                elapsed = time.perf_counter() - start_time
                
                # Get memory usage
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                # Calculate memory usage
                peak_mb = peak / 1024 / 1024
                if hasattr(data, 'element_size') and hasattr(data, 'numel'):
                    # PyTorch tensor
                    data_size_mb = (data.element_size() * data.numel()) / 1024 / 1024
                elif hasattr(data, 'nbytes'):
                    # NumPy array
                    data_size_mb = data.nbytes / 1024 / 1024
                else:
                    data_size_mb = peak_mb
                
                times.append(elapsed)
                memory_usage.append(data_size_mb)
                peak_memory_usage.append(peak_mb)
                
                del data
                gc.collect()
                
            except Exception as e:
                print(f"Error in {method_name}: {e}")
                return None
        
        if times:
            return {
                'mean': mean(times),
                'std': stdev(times) if len(times) > 1 else 0,
                'memory': mean(memory_usage),
                'peak_memory': mean(peak_memory_usage)
            }
        return None
    
    def _get_file_type(self, name: str) -> str:
        """Determine file type from name."""
        if 'multi_mef' in name:
            return 'multi_mef'
        elif 'mef' in name:
            return 'mef'
        elif 'table' in name:
            return 'table'
        elif 'scaled' in name:
            return 'scaled'
        elif 'wcs' in name:
            return 'wcs'
        elif 'compressed' in name:
            return 'compressed'
        elif 'timeseries' in name:
            return 'timeseries'
        else:
            return 'single'
    
    def _get_compression_type(self, name: str) -> str:
        """Determine compression type from name."""
        for comp in self.compression_types:
            if comp.lower() in name:
                return comp.lower()
        return 'uncompressed'
    
    def _astropy_read(self, filepath: Path, hdu_num: int):
        """Pure astropy read."""
        with astropy_fits.open(filepath, memmap=False) as hdul:
            return hdul[hdu_num].data.copy()
    
    def _astropy_to_torch(self, filepath: Path, hdu_num: int):
        """Astropy read + torch conversion."""
        with astropy_fits.open(filepath, memmap=False) as hdul:
            np_data = hdul[hdu_num].data.copy()
            if np_data.dtype.byteorder not in ('=', '|'):
                np_data = np_data.astype(np_data.dtype.newbyteorder('='))
            return torch.from_numpy(np_data)
    
    def _fitsio_to_torch(self, filepath: Path, hdu_num: int):
        """Fitsio read + torch conversion."""
        np_data = fitsio.read(str(filepath), ext=hdu_num)
        return torch.from_numpy(np_data)
    
    def generate_plots(self, results: List[Dict]):
        """Generate comprehensive plots from benchmark results."""
        print("\\nGenerating exhaustive plots...")
        df = pd.DataFrame(results)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Performance comparison by file type
        self._plot_performance_by_type(df)
        
        # 2. Memory usage analysis
        self._plot_memory_usage(df)
        
        # 3. Speedup analysis
        self._plot_speedup_analysis(df)
        
        # 4. Data type performance
        self._plot_data_type_performance(df)
        
        # 5. File size vs performance
        self._plot_size_performance(df)
        
        # 6. Compression analysis
        self._plot_compression_analysis(df)
        
        print(f"✓ Plots saved to {self.output_dir}")
    
    def _plot_performance_by_type(self, df):
        """Plot performance comparison by file type."""
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Comparison by File Type', fontsize=16)
        
        # Filter valid results
        methods = ['torchfits_mean', 'astropy_mean', 'fitsio_mean']
        
        for i, method in enumerate(methods):
            if method in df.columns:
                valid_df = df[df[method].notna()]
                if not valid_df.empty:
                    ax = axes[i//2, i%2]
                    sns.boxplot(data=valid_df, x='file_type', y=method, ax=ax)
                    ax.set_title(f'{method.replace("_mean", "").title()} Performance')
                    ax.set_ylabel('Time (seconds)')
                    plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Remove empty subplot
        if len(methods) < 4:
            axes[1, 1].remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_by_type.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_usage(self, df: pd.DataFrame):
        """Plot memory usage analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Memory Usage Analysis', fontsize=16)
        
        # Memory vs file size
        valid_df = df[df['torchfits_memory'].notna()]
        if not valid_df.empty:
            axes[0].scatter(valid_df['size_mb'], valid_df['torchfits_memory'], alpha=0.6)
            axes[0].plot([0, valid_df['size_mb'].max()], [0, valid_df['size_mb'].max()], 'r--', alpha=0.5)
            axes[0].set_xlabel('File Size (MB)')
            axes[0].set_ylabel('Memory Usage (MB)')
            axes[0].set_title('Memory Usage vs File Size')
        
        # Peak memory by data type
        if not valid_df.empty:
            sns.boxplot(data=valid_df, x='data_type', y='torchfits_peak_memory', ax=axes[1])
            axes[1].set_title('Peak Memory by Data Type')
            axes[1].set_ylabel('Peak Memory (MB)')
            plt.setp(axes[1].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_speedup_analysis(self, df: pd.DataFrame):
        """Plot speedup analysis."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create speedup comparison
        methods = ['astropy_mean', 'fitsio_mean', 'astropy_torch_mean', 'fitsio_torch_mean']
        speedups = {}
        
        for method in methods:
            if method in df.columns:
                valid_df = df[(df['torchfits_mean'].notna()) & (df[method].notna())]
                if not valid_df.empty:
                    speedup = valid_df[method] / valid_df['torchfits_mean']
                    speedups[method.replace('_mean', '')] = speedup
        
        if speedups:
            # Create box plot of speedups
            data_for_plot = []
            labels = []
            for method, speeds in speedups.items():
                data_for_plot.append(speeds)
                labels.append(method.replace('_', ' ').title())
            
            ax.boxplot(data_for_plot, labels=labels)
            ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No speedup')
            ax.set_ylabel('Speedup Factor (other/torchfits)')
            ax.set_title('torchfits Speedup vs Other Methods')
            ax.legend()
            plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'speedup_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_data_type_performance(self, df: pd.DataFrame):
        """Plot performance by data type."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        valid_df = df[df['torchfits_mean'].notna()]
        if not valid_df.empty:
            # Group by data type and dimensions
            perf_data = valid_df.groupby(['data_type', 'dimensions'])['torchfits_mean'].mean().reset_index()
            
            # Create heatmap
            pivot_data = perf_data.pivot(index='data_type', columns='dimensions', values='torchfits_mean')
            sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='viridis', ax=ax)
            ax.set_title('Average Performance by Data Type and Dimensions')
            ax.set_ylabel('Data Type')
            ax.set_xlabel('Dimensions')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'data_type_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_size_performance(self, df: pd.DataFrame):
        """Plot performance vs file size."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        valid_df = df[df['torchfits_mean'].notna()]
        if not valid_df.empty:
            # Color by file type
            file_types = valid_df['file_type'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(file_types)))
            
            for file_type, color in zip(file_types, colors):
                type_df = valid_df[valid_df['file_type'] == file_type]
                ax.scatter(type_df['size_mb'], type_df['torchfits_mean'], 
                          label=file_type, alpha=0.7, color=color)
            
            ax.set_xlabel('File Size (MB)')
            ax.set_ylabel('Performance (seconds)')
            ax.set_title('Performance vs File Size by Type')
            ax.legend()
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'size_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_compression_analysis(self, df: pd.DataFrame):
        """Plot compression analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Compression Analysis', fontsize=16)
        
        # Performance by compression type
        valid_df = df[df['torchfits_mean'].notna()]
        if not valid_df.empty:
            sns.boxplot(data=valid_df, x='compression', y='torchfits_mean', ax=axes[0])
            axes[0].set_title('Performance by Compression Type')
            axes[0].set_ylabel('Time (seconds)')
            plt.setp(axes[0].get_xticklabels(), rotation=45)
            
            # File size reduction
            if len(valid_df['compression'].unique()) > 1:
                size_by_comp = valid_df.groupby('compression')['size_mb'].mean()
                axes[1].bar(size_by_comp.index, size_by_comp.values)
                axes[1].set_title('Average File Size by Compression')
                axes[1].set_ylabel('File Size (MB)')
                plt.setp(axes[1].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'compression_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, results: List[Dict]):
        """Generate a comprehensive summary report."""
        print("\\nGenerating exhaustive summary report...")
        
        df = pd.DataFrame(results)
        
        with open(self.summary_file, 'w') as f:
            f.write("# torchfits Exhaustive Benchmark Report\\n\\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            # System information
            f.write("## System Information\\n\\n")
            f.write(f"- Python: {sys.version.split()[0]}\\n")
            f.write(f"- PyTorch: {torch.__version__}\\n")
            f.write(f"- CUDA available: {torch.cuda.is_available()}\\n")
            if torch.cuda.is_available():
                f.write(f"- CUDA device: {torch.cuda.get_device_name()}\\n")
            f.write(f"- astropy available: True\\n")
            f.write(f"- fitsio available: True\\n")
            f.write(f"- System memory: {psutil.virtual_memory().total / (1024**3):.1f} GB\\n")
            f.write("\\n")
            
            # Test coverage summary
            f.write("## Test Coverage Summary\\n\\n")
            f.write(f"- Total test files: {len(results)}\\n")
            f.write(f"- File types tested: {', '.join(sorted(df['file_type'].unique()))}\\n")
            f.write(f"- Data types tested: {', '.join(sorted(df['data_type'].unique()))}\\n")
            f.write(f"- Dimensions tested: {', '.join(sorted(df['dimensions'].unique()))}\\n")
            f.write(f"- Compression types: {', '.join(sorted(df['compression'].unique()))}\\n")
            f.write("\\n")
            
            # Performance summary
            if 'torchfits_mean' in df.columns:
                f.write("## Performance Summary\\n\\n")
                valid_df = df[df['torchfits_mean'].notna()]
                
                f.write(f"- Fastest torchfits time: {valid_df['torchfits_mean'].min():.6f}s\\n")
                f.write(f"- Slowest torchfits time: {valid_df['torchfits_mean'].max():.6f}s\\n")
                f.write(f"- Average torchfits time: {valid_df['torchfits_mean'].mean():.6f}s\\n")
                f.write(f"- Median torchfits time: {valid_df['torchfits_mean'].median():.6f}s\\n")
                f.write("\\n")
            
            # File type breakdown
            f.write("## Performance by File Type\\n\\n")
            if 'torchfits_mean' in df.columns:
                type_stats = df.groupby('file_type')['torchfits_mean'].agg(['count', 'mean', 'min', 'max']).round(6)
                f.write(type_stats.to_string())
                f.write("\\n\\n")
            
            # Ranking analysis
            if 'torchfits_rank' in df.columns:
                f.write("## Ranking Analysis\\n\\n")
                rank_counts = df['torchfits_rank'].value_counts().sort_index()
                total_valid = rank_counts.sum()
                
                f.write(f"- Times torchfits ranked #1: {rank_counts.get(1, 0)} ({rank_counts.get(1, 0)/total_valid*100:.1f}%)\\n")
                f.write(f"- Times torchfits ranked #2: {rank_counts.get(2, 0)} ({rank_counts.get(2, 0)/total_valid*100:.1f}%)\\n")
                f.write(f"- Times torchfits ranked #3+: {sum(rank_counts[rank_counts.index >= 3])} ({sum(rank_counts[rank_counts.index >= 3])/total_valid*100:.1f}%)\\n")
                f.write(f"- Average ranking: {df['torchfits_rank'].mean():.2f}\\n")
                f.write("\\n")
            
            # Memory analysis
            if 'torchfits_memory' in df.columns:
                f.write("## Memory Analysis\\n\\n")
                mem_df = df[df['torchfits_memory'].notna()]
                if not mem_df.empty:
                    f.write(f"- Average memory usage: {mem_df['torchfits_memory'].mean():.1f} MB\\n")
                    f.write(f"- Peak memory usage: {mem_df['torchfits_peak_memory'].mean():.1f} MB\\n")
                    f.write(f"- Memory efficiency (data/peak): {(mem_df['torchfits_memory']/mem_df['torchfits_peak_memory']).mean():.2f}\\n")
                f.write("\\n")
            
            # Top performers
            f.write("## Best Performing Files\\n\\n")
            if 'torchfits_mean' in df.columns:
                fastest = valid_df.nsmallest(10, 'torchfits_mean')[['filename', 'torchfits_mean', 'size_mb', 'file_type']]
                f.write("### Fastest Files:\\n")
                for _, row in fastest.iterrows():
                    f.write(f"- {row['filename']}: {row['torchfits_mean']:.6f}s ({row['size_mb']:.2f} MB, {row['file_type']})\\n")
                f.write("\\n")
            
            # Issues and failures
            failed_files = df[df['torchfits_mean'].isna()]
            if not failed_files.empty:
                f.write("## Failed Tests\\n\\n")
                for _, row in failed_files.iterrows():
                    f.write(f"- {row['filename']}: Failed to benchmark\\n")
                f.write("\\n")
            
            # Comprehensive recommendations
            f.write("## Comprehensive Recommendations\\n\\n")
            f.write("Based on the exhaustive benchmark results:\\n\\n")
            
            if 'torchfits_rank' in df.columns:
                avg_rank = df['torchfits_rank'].mean()
                if avg_rank <= 2:
                    f.write("✅ **torchfits shows excellent performance** across all test scenarios\\n")
                elif avg_rank <= 3:
                    f.write("⚠️ **torchfits shows good performance** with opportunities for optimization\\n")
                else:
                    f.write("❌ **torchfits performance needs significant improvement**\\n")
            
            # Specific findings
            f.write("\\n### Specific Findings:\\n\\n")
            
            # Best file types
            if 'torchfits_rank' in df.columns and 'file_type' in df.columns:
                best_types = df.groupby('file_type')['torchfits_rank'].mean().sort_values()
                f.write(f"- **Best file types**: {', '.join(best_types.head(3).index)}\\n")
                f.write(f"- **Challenging file types**: {', '.join(best_types.tail(3).index)}\\n")
            
            # Data type performance
            if 'data_type' in df.columns and 'torchfits_mean' in df.columns:
                dtype_perf = df.groupby('data_type')['torchfits_mean'].mean().sort_values()
                f.write(f"- **Fastest data types**: {', '.join(dtype_perf.head(3).index)}\\n")
                f.write(f"- **Slowest data types**: {', '.join(dtype_perf.tail(3).index)}\\n")
            
            f.write("\\n")
            f.write("## Files Generated\\n\\n")
            f.write(f"- Detailed results: `{self.csv_file.name}`\\n")
            f.write(f"- Performance plots: `*.png` files\\n")
            f.write(f"- This summary: `{self.summary_file.name}`\\n")
            f.write("\\n")
            f.write("## Next Steps\\n\\n")
            f.write("1. Review detailed CSV results for specific performance bottlenecks\\n")
            f.write("2. Examine plots for visual performance patterns\\n")
            f.write("3. Focus optimization efforts on underperforming file types\\n")
            f.write("4. Consider implementing specialized paths for best-performing scenarios\\n")
        
        print(f"✓ Exhaustive summary report saved to: {self.summary_file}")
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print(f"✓ Cleaned up temporary directory: {self.temp_dir}")
    
    def run_full_suite(self):
        """Run the complete exhaustive benchmark suite."""
        try:
            print("Starting exhaustive torchfits benchmark suite...")
            print(f"Output directory: {self.output_dir}")
            
            # Create test files
            files = self.create_test_files()
            
            # Run benchmarks
            results = self.run_exhaustive_benchmarks(files)
            
            # Generate visualizations
            self.generate_plots(results)
            
            # Generate summary report
            self.generate_summary_report(results)
            
            print("\\n" + "=" * 80)
            print("EXHAUSTIVE BENCHMARK SUITE COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"Results saved to: {self.output_dir}")
            print(f"- CSV data: {self.csv_file}")
            print(f"- Summary: {self.summary_file}")
            print(f"- Plots: {self.output_dir}/*.png")
            
        finally:
            self.cleanup()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Exhaustive torchfits benchmark suite')
    parser.add_argument('--output-dir', type=Path, default=Path('benchmark_results'),
                       help='Output directory for results')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Keep temporary files for debugging')
    
    args = parser.parse_args()
    
    suite = ExhaustiveBenchmarkSuite(output_dir=args.output_dir)
    
    if args.no_cleanup:
        # Override cleanup method
        suite.cleanup = lambda: print(f"Temporary files kept in: {suite.temp_dir}")
    
    suite.run_full_suite()


if __name__ == "__main__":
    main()
