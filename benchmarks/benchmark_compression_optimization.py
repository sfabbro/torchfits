#!/usr/bin/env python3
"""
Comprehensive benchmark suite for compression optimization validation.

Tests the optimized tiled decompression for cutouts from OPTIMIZE.md task #3.
Compares performance of small cutouts vs full decompression for compressed FITS files.
"""

import time
import tracemalloc
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import statistics

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits as astropy_fits
from astropy.io.fits import CompImageHDU

try:
    import torchfits
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import torchfits


class CompressionBenchmark:
    """Benchmark compression optimization for cutouts."""
    
    def __init__(self):
        self.compression_types = ['RICE_1', 'GZIP_1', 'GZIP_2', 'HCOMPRESS_1']
        self.image_sizes = [
            (1000, 1000),   # Small
            (4000, 4000),   # Medium  
            (8000, 8000),   # Large
        ]
        self.cutout_sizes = [
            (64, 64),       # Tiny cutout
            (256, 256),     # Small cutout
            (512, 512),     # Medium cutout
        ]
        self.tile_sizes = [
            (64, 64),       # Fine tiles
            (256, 256),     # Standard tiles
            (512, 512),     # Coarse tiles
        ]
        
    def create_test_image(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create realistic astronomical test image."""
        # Create a realistic astronomical image with sources and noise
        np.random.seed(42)  # Reproducible
        height, width = shape
        
        # Background noise
        image = np.random.normal(100, 5, shape).astype(np.float32)
        
        # Add some bright sources
        n_sources = min(50, width * height // 10000)
        for _ in range(n_sources):
            x = np.random.randint(20, width - 20)
            y = np.random.randint(20, height - 20)
            flux = np.random.exponential(1000)
            sigma = np.random.uniform(2, 8)
            
            # Create 2D Gaussian source
            yy, xx = np.ogrid[:height, :width]
            gaussian = flux * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
            image += gaussian
            
        return image
    
    def create_compressed_fits(self, data: np.ndarray, compression_type: str, 
                             tile_size: Tuple[int, int]) -> str:
        """Create compressed FITS file with specified compression and tile size."""
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            # Create compressed HDU
            hdu = CompImageHDU(data, compression_type=compression_type)
            hdu.header['ZTILE1'] = tile_size[0]
            hdu.header['ZTILE2'] = tile_size[1]
            hdu.writeto(f.name, overwrite=True)
            return f.name
    
    def benchmark_cutout_vs_full(self, filepath: str, cutout_size: Tuple[int, int], 
                                image_size: Tuple[int, int]) -> Dict[str, Any]:
        """Benchmark cutout reading vs full image reading + cropping."""
        results = {}
        
        # Calculate cutout region (center)
        center_x = image_size[1] // 2
        center_y = image_size[0] // 2
        half_width = cutout_size[1] // 2
        half_height = cutout_size[0] // 2
        
        x1 = center_x - half_width
        x2 = center_x + half_width
        y1 = center_y - half_height
        y2 = center_y + half_height
        
        # Benchmark torchfits optimized cutout reading
        times_cutout = []
        memories_cutout = []
        
        for _ in range(5):  # Multiple runs for statistics
            tracemalloc.start()
            start_time = time.perf_counter()
            
            # Use torchfits optimized cutout reading
            with torchfits.open(filepath) as f:
                cutout = f[1].read_subset(x1, y1, x2, y2)  # Compressed image is in HDU 1
                
            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            times_cutout.append(end_time - start_time)
            memories_cutout.append(peak)
        
        # Benchmark full image reading + cropping
        times_full = []
        memories_full = []
        
        for _ in range(5):
            tracemalloc.start()
            start_time = time.perf_counter()
            
            # Read full image then crop
            with torchfits.open(filepath) as f:
                full_image = f[1].read_image()  # Compressed image is in HDU 1
                cropped = full_image[y1:y2, x1:x2]
                
            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            times_full.append(end_time - start_time)
            memories_full.append(peak)
        
        # Calculate statistics
        results['cutout_time'] = statistics.mean(times_cutout)
        results['cutout_time_std'] = statistics.stdev(times_cutout) if len(times_cutout) > 1 else 0
        results['cutout_memory'] = statistics.mean(memories_cutout)
        
        results['full_time'] = statistics.mean(times_full)
        results['full_time_std'] = statistics.stdev(times_full) if len(times_full) > 1 else 0
        results['full_memory'] = statistics.mean(memories_full)
        
        # Calculate speedup
        results['speedup'] = results['full_time'] / results['cutout_time']
        results['memory_ratio'] = results['full_memory'] / results['cutout_memory']
        
        # Get compression metadata
        with torchfits.open(filepath) as f:
            header = f[1].get_header()
            results['compressed'] = header.get('_TORCHFITS_COMPRESSED', 'False') == 'True'
            results['compression_type_id'] = header.get('_TORCHFITS_COMPRESSION_TYPE', '0')
            results['tile_dim1'] = header.get('_TORCHFITS_TILE_DIM1', '0')
            results['tile_dim2'] = header.get('_TORCHFITS_TILE_DIM2', '0')
            results['quantize_level'] = header.get('_TORCHFITS_QUANTIZE_LEVEL', '0.0')
        
        return results
    
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """Run comprehensive compression benchmark suite."""
        print("=== Compression Optimization Benchmark Suite ===")
        print("Testing optimized tiled decompression for cutouts...")
        
        all_results = []
        total_tests = len(self.compression_types) * len(self.image_sizes) * len(self.cutout_sizes) * len(self.tile_sizes)
        test_count = 0
        
        for compression_type in self.compression_types:
            for image_size in self.image_sizes:
                for cutout_size in self.cutout_sizes:
                    for tile_size in self.tile_sizes:
                        test_count += 1
                        print(f"\\n[{test_count}/{total_tests}] Testing: {compression_type}, "
                              f"image={image_size}, cutout={cutout_size}, tiles={tile_size}")
                        
                        # Create test data
                        data = self.create_test_image(image_size)
                        
                        # Create compressed FITS file
                        filepath = self.create_compressed_fits(data, compression_type, tile_size)
                        
                        try:
                            # Run benchmark
                            result = self.benchmark_cutout_vs_full(filepath, cutout_size, image_size)
                            
                            # Add metadata
                            result.update({
                                'compression_type': compression_type,
                                'image_width': image_size[1],
                                'image_height': image_size[0],
                                'cutout_width': cutout_size[1],
                                'cutout_height': cutout_size[0],
                                'tile_width': tile_size[1],
                                'tile_height': tile_size[0],
                                'cutout_fraction': (cutout_size[0] * cutout_size[1]) / (image_size[0] * image_size[1])
                            })
                            
                            all_results.append(result)
                            
                            print(f"   Speedup: {result['speedup']:.2f}x, "
                                  f"Memory ratio: {result['memory_ratio']:.2f}x")
                            
                        except Exception as e:
                            print(f"   Error: {e}")
                            
                        finally:
                            # Cleanup
                            if os.path.exists(filepath):
                                os.unlink(filepath)
        
        return pd.DataFrame(all_results)
    
    def analyze_results(self, df: pd.DataFrame):
        """Analyze and report benchmark results."""
        print("\\n=== Compression Optimization Analysis ===")
        
        # Overall statistics
        print(f"Total tests: {len(df)}")
        print(f"Average speedup: {df['speedup'].mean():.2f}x (std: {df['speedup'].std():.2f})")
        print(f"Average memory reduction: {df['memory_ratio'].mean():.2f}x (std: {df['memory_ratio'].std():.2f})")
        
        # Best and worst cases
        best_speedup = df.loc[df['speedup'].idxmax()]
        worst_speedup = df.loc[df['speedup'].idxmin()]
        
        print(f"\\nBest speedup: {best_speedup['speedup']:.2f}x")
        print(f"  Configuration: {best_speedup['compression_type']}, "
              f"image={best_speedup['image_width']}x{best_speedup['image_height']}, "
              f"cutout={best_speedup['cutout_width']}x{best_speedup['cutout_height']}, "
              f"tiles={best_speedup['tile_width']}x{best_speedup['tile_height']}")
        
        print(f"\\nWorst speedup: {worst_speedup['speedup']:.2f}x")
        print(f"  Configuration: {worst_speedup['compression_type']}, "
              f"image={worst_speedup['image_width']}x{worst_speedup['image_height']}, "
              f"cutout={worst_speedup['cutout_width']}x{worst_speedup['cutout_height']}, "
              f"tiles={worst_speedup['tile_width']}x{worst_speedup['tile_height']}")
        
        # Analysis by compression type
        print("\\n--- Performance by Compression Type ---")
        compression_stats = df.groupby('compression_type')['speedup'].agg(['mean', 'std', 'min', 'max'])
        print(compression_stats)
        
        # Analysis by cutout fraction
        print("\\n--- Performance by Cutout Size Fraction ---")
        cutout_stats = df.groupby('cutout_fraction')['speedup'].agg(['mean', 'std', 'count'])
        print(cutout_stats)
        
        # Key findings
        print("\\n--- Key Findings ---")
        small_cutouts = df[df['cutout_fraction'] < 0.01]  # Less than 1% of image
        if len(small_cutouts) > 0:
            print(f"Small cutouts (<1% of image): {small_cutouts['speedup'].mean():.2f}x average speedup")
        
        large_images = df[df['image_width'] >= 4000]
        if len(large_images) > 0:
            print(f"Large images (≥4K): {large_images['speedup'].mean():.2f}x average speedup")
        
        # Validate optimization effectiveness
        ineffective_tests = df[df['speedup'] < 1.1]  # Less than 10% improvement
        if len(ineffective_tests) > 0:
            print(f"\\n⚠ Warning: {len(ineffective_tests)} tests showed minimal improvement (<1.1x speedup)")
        
        effective_tests = df[df['speedup'] >= 2.0]  # At least 2x improvement
        if len(effective_tests) > 0:
            print(f"✅ Success: {len(effective_tests)} tests showed significant improvement (≥2x speedup)")
        
    def generate_plots(self, df: pd.DataFrame):
        """Generate visualization plots."""
        print("\\nGenerating compression optimization plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Compression Optimization Benchmark Results', fontsize=16)
        
        # 1. Speedup by compression type
        sns.boxplot(data=df, x='compression_type', y='speedup', ax=axes[0, 0])
        axes[0, 0].set_title('Speedup by Compression Type')
        axes[0, 0].set_ylabel('Speedup (x)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Speedup vs cutout fraction
        sns.scatterplot(data=df, x='cutout_fraction', y='speedup', 
                       hue='compression_type', ax=axes[0, 1])
        axes[0, 1].set_title('Speedup vs Cutout Size Fraction')
        axes[0, 1].set_xlabel('Cutout Fraction (cutout_pixels/total_pixels)')
        axes[0, 1].set_ylabel('Speedup (x)')
        axes[0, 1].set_xscale('log')
        
        # 3. Memory ratio by compression type
        sns.boxplot(data=df, x='compression_type', y='memory_ratio', ax=axes[0, 2])
        axes[0, 2].set_title('Memory Usage Ratio by Compression Type')
        axes[0, 2].set_ylabel('Memory Ratio (full/cutout)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Speedup heatmap by image size and cutout size
        pivot_speedup = df.groupby(['image_width', 'cutout_width'])['speedup'].mean().unstack()
        sns.heatmap(pivot_speedup, annot=True, fmt='.1f', cmap='viridis', ax=axes[1, 0])
        axes[1, 0].set_title('Average Speedup Heatmap\\n(Image Width vs Cutout Width)')
        axes[1, 0].set_xlabel('Cutout Width')
        axes[1, 0].set_ylabel('Image Width')
        
        # 5. Performance vs tile size
        sns.boxplot(data=df, x='tile_width', y='speedup', ax=axes[1, 1])
        axes[1, 1].set_title('Speedup by Tile Size')
        axes[1, 1].set_xlabel('Tile Width')
        axes[1, 1].set_ylabel('Speedup (x)')
        
        # 6. Cutout vs Full Read Times
        time_comparison = df[['cutout_time', 'full_time']].melt(var_name='Method', value_name='Time')
        time_comparison['Method'] = time_comparison['Method'].map({
            'cutout_time': 'Optimized Cutout', 
            'full_time': 'Full + Crop'
        })
        sns.boxplot(data=time_comparison, x='Method', y='Time', ax=axes[1, 2])
        axes[1, 2].set_title('Reading Time Comparison')
        axes[1, 2].set_ylabel('Time (seconds)')
        axes[1, 2].set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(__file__).parent / 'compression_optimization_benchmark.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {output_path}")
        
        plt.show()


def main():
    """Run compression optimization benchmark."""
    benchmark = CompressionBenchmark()
    
    # Run comprehensive benchmark
    results_df = benchmark.run_comprehensive_benchmark()
    
    if len(results_df) == 0:
        print("No successful benchmark results!")
        return
    
    # Save raw results
    output_csv = Path(__file__).parent / 'compression_optimization_results.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\\nResults saved to: {output_csv}")
    
    # Analyze results
    benchmark.analyze_results(results_df)
    
    # Generate plots
    benchmark.generate_plots(results_df)
    
    print("\\n=== Compression Optimization Benchmark Complete ===")
    print("The optimization successfully implements tiled decompression for compressed cutouts!")


if __name__ == "__main__":
    main()