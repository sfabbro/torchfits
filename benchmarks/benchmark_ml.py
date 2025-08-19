"""
Machine Learning workflow benchmarks for torchfits.

Benchmarks DataLoader performance, batch processing, and GPU transfer
for different dataset sizes and configurations.
"""

import time
import sys
import os
from pathlib import Path
import numpy as np
import torch
import tempfile
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import torchfits
from torchfits.datasets import FITSDataset
from torchfits.dataloader import create_dataloader

try:
    from astropy.io import fits as astropy_fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

class MLBenchmarkData:
    """Generate ML-focused test data."""
    
    @staticmethod
    def create_image_dataset(num_files, image_shape=(256, 256), temp_dir=None):
        """Create a dataset of FITS image files."""
        if not HAS_ASTROPY:
            return []
        
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        files = []
        for i in range(num_files):
            data = np.random.randn(*image_shape).astype(np.float32)
            filename = Path(temp_dir) / f"image_{i:04d}.fits"
            
            hdu = astropy_fits.PrimaryHDU(data)
            hdul = astropy_fits.HDUList([hdu])
            hdul.writeto(filename, overwrite=True)
            files.append(str(filename))
        
        return files
    
    @staticmethod
    def create_mixed_dataset(num_files, temp_dir=None):
        """Create dataset with mixed data types and sizes."""
        if not HAS_ASTROPY:
            return []
        
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        files = []
        shapes = [(128, 128), (256, 256), (512, 512)]
        
        for i in range(num_files):
            shape = shapes[i % len(shapes)]
            data = np.random.randn(*shape).astype(np.float32)
            filename = Path(temp_dir) / f"mixed_{i:04d}.fits"
            
            hdu = astropy_fits.PrimaryHDU(data)
            hdul = astropy_fits.HDUList([hdu])
            hdul.writeto(filename, overwrite=True)
            files.append(str(filename))
        
        return files

class MLBenchmarkSuite:
    """ML workflow benchmark suite."""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.results = {}
    
    def benchmark_dataset_creation(self):
        """Benchmark FITSDataset creation and indexing."""
        print("=== Dataset Creation Benchmarks ===")
        
        dataset_sizes = [10, 50, 100, 500]
        
        for size in dataset_sizes:
            print(f"\nDataset size: {size} files")
            
            # Create test files
            files = MLBenchmarkData.create_image_dataset(size, temp_dir=self.temp_dir)
            if not files:
                print("Skipping (no astropy)")
                continue
            
            # Benchmark dataset creation
            start_time = time.perf_counter()
            dataset = FITSDataset(files)
            creation_time = time.perf_counter() - start_time
            print(f"  Creation: {creation_time:.4f}s")
            
            # Benchmark random access
            indices = np.random.randint(0, len(dataset), 10)
            start_time = time.perf_counter()
            for idx in indices:
                try:
                    sample = dataset[idx]
                except Exception as e:
                    print(f"  Access failed: {e}")
                    break
            access_time = time.perf_counter() - start_time
            print(f"  Random access (10 samples): {access_time:.4f}s")
    
    def benchmark_dataloader_performance(self):
        """Benchmark DataLoader performance with different configurations."""
        print("\n=== DataLoader Performance Benchmarks ===")
        
        # Create test dataset
        num_files = 100
        files = MLBenchmarkData.create_image_dataset(num_files, temp_dir=self.temp_dir)
        if not files:
            print("Skipping DataLoader benchmarks (no astropy)")
            return
        
        batch_sizes = [1, 4, 16, 32]
        # Note: For small datasets, multiprocessing workers add overhead
        # Workers > 0 spawn processes which is slower for small file counts
        num_workers_list = [0, 1]  # Reduced from [0, 1, 2, 4] to avoid excessive overhead
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            
            for num_workers in num_workers_list:
                print(f"  Workers: {num_workers}")
                
                try:
                    # Create DataLoader
                    dataset = FITSDataset(files)
                    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=False
                    )
                    
                    # Benchmark iteration
                    start_time = time.perf_counter()
                    total_samples = 0
                    
                    for batch in dataloader:
                        if isinstance(batch, torch.Tensor):
                            total_samples += batch.shape[0]
                        else:
                            total_samples += len(batch)
                        
                        # Only process first few batches for timing
                        if total_samples >= 50:
                            break
                    
                    iteration_time = time.perf_counter() - start_time
                    throughput = total_samples / iteration_time if iteration_time > 0 else 0
                    
                    print(f"    Time: {iteration_time:.4f}s")
                    print(f"    Throughput: {throughput:.1f} samples/sec")
                    
                except Exception as e:
                    print(f"    FAILED: {e}")
    
    def benchmark_gpu_transfer(self):
        """Benchmark GPU data transfer performance."""
        print("\n=== GPU Transfer Benchmarks ===")
        
        if not torch.cuda.is_available():
            print("CUDA not available, skipping GPU benchmarks")
            return
        
        # Create test data
        files = MLBenchmarkData.create_image_dataset(20, temp_dir=self.temp_dir)
        if not files:
            print("Skipping GPU benchmarks (no astropy)")
            return
        
        dataset = FITSDataset(files)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        print("GPU transfer performance:")
        
        try:
            total_transfer_time = 0
            total_samples = 0
            
            for batch in dataloader:
                if isinstance(batch, torch.Tensor):
                    # Time GPU transfer
                    start_time = time.perf_counter()
                    gpu_batch = batch.cuda()
                    transfer_time = time.perf_counter() - start_time
                    
                    total_transfer_time += transfer_time
                    total_samples += batch.shape[0]
                    
                    # Test a few batches
                    if total_samples >= 16:
                        break
            
            avg_transfer_time = total_transfer_time / total_samples
            throughput = total_samples / total_transfer_time
            
            print(f"  Average transfer time: {avg_transfer_time:.6f}s per sample")
            print(f"  Transfer throughput: {throughput:.1f} samples/sec")
            
        except Exception as e:
            print(f"GPU transfer failed: {e}")
    
    def benchmark_memory_efficiency(self):
        """Benchmark memory usage patterns."""
        print("\n=== Memory Efficiency Benchmarks ===")
        
        # Test different image sizes
        image_sizes = [(64, 64), (256, 256), (512, 512), (1024, 1024)]
        
        for size in image_sizes:
            print(f"\nImage size: {size[0]}x{size[1]}")
            
            # Create small dataset
            files = MLBenchmarkData.create_image_dataset(10, image_shape=size, temp_dir=self.temp_dir)
            if not files:
                continue
            
            try:
                dataset = FITSDataset(files)
                dataloader = DataLoader(dataset, batch_size=4)
                
                # Measure memory usage during iteration
                start_time = time.perf_counter()
                peak_memory = 0
                
                for i, batch in enumerate(dataloader):
                    if isinstance(batch, torch.Tensor):
                        # Estimate memory usage
                        batch_memory = batch.numel() * batch.element_size()
                        peak_memory = max(peak_memory, batch_memory)
                    
                    if i >= 2:  # Test a few batches
                        break
                
                iteration_time = time.perf_counter() - start_time
                
                print(f"  Peak batch memory: {peak_memory / 1024 / 1024:.1f} MB")
                print(f"  Iteration time: {iteration_time:.4f}s")
                
            except Exception as e:
                print(f"  Memory test failed: {e}")
    
    def benchmark_transform_pipeline(self):
        """Benchmark data transformation pipeline."""
        print("\n=== Transform Pipeline Benchmarks ===")
        
        from torchfits.transforms import AsinhStretch, Normalize, RandomCrop, Compose
        
        # Create test data
        files = MLBenchmarkData.create_image_dataset(20, image_shape=(512, 512), temp_dir=self.temp_dir)
        if not files:
            print("Skipping transform benchmarks (no astropy)")
            return
        
        # Define transform pipelines
        transforms = {
            'none': None,
            'normalize': Normalize(),
            'asinh': AsinhStretch(),
            'crop': RandomCrop(256),
            'full_pipeline': Compose([
                RandomCrop(256),
                AsinhStretch(),
                Normalize()
            ])
        }
        
        for transform_name, transform in transforms.items():
            print(f"\nTransform: {transform_name}")
            
            try:
                dataset = FITSDataset(files, transform=transform)
                dataloader = DataLoader(dataset, batch_size=4)
                
                start_time = time.perf_counter()
                total_samples = 0
                
                for batch in dataloader:
                    if isinstance(batch, torch.Tensor):
                        total_samples += batch.shape[0]
                    
                    if total_samples >= 16:
                        break
                
                transform_time = time.perf_counter() - start_time
                throughput = total_samples / transform_time
                
                print(f"  Time: {transform_time:.4f}s")
                print(f"  Throughput: {throughput:.1f} samples/sec")
                
            except Exception as e:
                print(f"  Transform failed: {e}")
    
    def run_all_benchmarks(self):
        """Run complete ML benchmark suite."""
        print("torchfits ML Workflow Benchmark Suite")
        print("="*45)
        
        if not HAS_ASTROPY:
            print("WARNING: astropy not available, most benchmarks will be skipped")
        
        self.benchmark_dataset_creation()
        self.benchmark_dataloader_performance()
        self.benchmark_gpu_transfer()
        self.benchmark_memory_efficiency()
        self.benchmark_transform_pipeline()

def main():
    """Run ML benchmark suite."""
    suite = MLBenchmarkSuite()
    suite.run_all_benchmarks()

if __name__ == "__main__":
    main()