"""
Buffer management benchmarks for torchfits.

This module provides focused benchmarks for buffer management performance,
including memory pools, streaming buffers, and allocation patterns.
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from torchfits import buffer


class BufferBenchmark:
    """Benchmark suite for buffer management performance."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Test configurations
        self.pool_sizes = [64, 128, 256, 512]  # MB
        self.buffer_shapes = [(100, 100), (500, 500), (1000, 1000)]
        self.streaming_capacities = [10, 50, 100, 500]
    
    def run_memory_pool_benchmarks(self) -> Dict[str, Any]:
        """Benchmark memory pool performance."""
        print("  🗄️  Testing memory pool performance...")
        
        results = {}
        
        for pool_size in self.pool_sizes:
            buffer.configure_buffers(
                buffer_size_mb=pool_size,
                num_buffers=8,
                enable_memory_pool=True
            )
            
            pool_results = {}
            
            for shape in self.buffer_shapes:
                shape_key = f"{shape[0]}x{shape[1]}"
                
                # Benchmark buffer allocation
                start_time = time.time()
                buffers = []
                for i in range(20):
                    buf = buffer.get_buffer_manager().get_buffer(
                        f"test_{i}", shape, torch.float32, 'cpu'
                    )
                    buffers.append(buf)
                
                alloc_time = time.time() - start_time
                
                # Benchmark buffer access
                start_time = time.time()
                for buf in buffers:
                    buf.fill_(1.0)
                    _ = buf.mean()
                
                access_time = time.time() - start_time
                
                pool_results[shape_key] = {
                    'allocation_time_ms': alloc_time * 1000,
                    'access_time_ms': access_time * 1000,
                    'memory_usage_mb': buffer.get_buffer_stats()['total_memory_mb']
                }
                
                print(f"    Pool {pool_size}MB, {shape_key}: "
                      f"alloc={alloc_time*1000:.2f}ms, access={access_time*1000:.2f}ms")
                
                # Clean up
                buffer.clear_buffers()
            
            results[f"pool_{pool_size}mb"] = pool_results
        
        return results
    
    def run_streaming_buffer_benchmarks(self) -> Dict[str, Any]:
        """Benchmark streaming buffer performance."""
        print("  🔄 Testing streaming buffer performance...")
        
        results = {}
        
        for capacity in self.streaming_capacities:
            stream_buf = buffer.create_streaming_buffer(
                f"stream_{capacity}", capacity, (64, 64), torch.float32
            )
            
            # Benchmark put/get operations
            test_data = [torch.randn(64, 64) for _ in range(capacity * 2)]
            
            start_time = time.time()
            for i, data in enumerate(test_data):
                if i < capacity:
                    stream_buf.put(data)
                else:
                    stream_buf.get()
                    stream_buf.put(data)
            
            streaming_time = time.time() - start_time
            
            results[f"capacity_{capacity}"] = {
                'throughput_ops_per_sec': len(test_data) / streaming_time,
                'avg_time_per_op_us': (streaming_time / len(test_data)) * 1e6
            }
            
            print(f"    Capacity {capacity}: {len(test_data) / streaming_time:.1f} ops/sec")
        
        return results
    
    def run_workload_optimization_benchmarks(self) -> Dict[str, Any]:
        """Benchmark workload optimization."""
        print("  ⚙️  Testing workload optimization...")
        
        results = {}
        
        workloads = [
            {'file_size_mb': 10.0, 'concurrent_files': 2, 'name': 'small'},
            {'file_size_mb': 50.0, 'concurrent_files': 4, 'name': 'medium'},
            {'file_size_mb': 100.0, 'concurrent_files': 8, 'name': 'large'}
        ]
        
        for workload in workloads:
            # Test optimization time
            start_time = time.time()
            buffer.optimize_for_workload(
                workload['file_size_mb'], 
                workload['concurrent_files']
            )
            optimization_time = time.time() - start_time
            
            # Get optimized configuration
            stats = buffer.get_buffer_stats()
            
            results[workload['name']] = {
                'optimization_time_ms': optimization_time * 1000,
                'buffer_size_mb': stats.get('buffer_size_mb', 0),
                'num_buffers': stats.get('num_buffers_config', 0),
                'total_memory_mb': stats.get('total_memory_mb', 0)
            }
            
            print(f"    {workload['name'].title()} workload: "
                  f"optimized in {optimization_time*1000:.2f}ms")
        
        return results
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """Run all buffer benchmarks."""
        print("🗄️  Running Buffer Management Benchmarks...")
        
        results = {
            'memory_pools': self.run_memory_pool_benchmarks(),
            'streaming_buffers': self.run_streaming_buffer_benchmarks(),
            'workload_optimization': self.run_workload_optimization_benchmarks()
        }
        
        return results
    
    def generate_plots(self, results: Dict[str, Any]):
        """Generate buffer performance plots."""
        print("  📊 Generating buffer plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Buffer Management Performance', fontsize=16)
        
        # Memory pool allocation times
        if 'memory_pools' in results:
            pool_data = []
            for pool_name, pool_results in results['memory_pools'].items():
                pool_size = pool_name.replace('pool_', '').replace('mb', '')
                for shape_key, metrics in pool_results.items():
                    pool_data.append({
                        'pool_size': f"{pool_size}MB",
                        'shape': shape_key,
                        'allocation_time': metrics['allocation_time_ms']
                    })
            
            if pool_data:
                df_pool = pd.DataFrame(pool_data)
                pivot_pool = df_pool.pivot(index='shape', columns='pool_size', values='allocation_time')
                pivot_pool.plot(kind='bar', ax=axes[0, 0])
                axes[0, 0].set_title('Memory Pool Allocation Time')
                axes[0, 0].set_ylabel('Time (ms)')
                axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Streaming buffer throughput
        if 'streaming_buffers' in results:
            stream_data = []
            for capacity_key, metrics in results['streaming_buffers'].items():
                capacity = capacity_key.replace('capacity_', '')
                stream_data.append({
                    'capacity': int(capacity),
                    'throughput': metrics['throughput_ops_per_sec']
                })
            
            if stream_data:
                df_stream = pd.DataFrame(stream_data)
                axes[0, 1].plot(df_stream['capacity'], df_stream['throughput'], marker='o')
                axes[0, 1].set_title('Streaming Buffer Throughput')
                axes[0, 1].set_xlabel('Buffer Capacity')
                axes[0, 1].set_ylabel('Throughput (ops/sec)')
                axes[0, 1].grid(True, alpha=0.3)
        
        # Workload optimization
        if 'workload_optimization' in results:
            workload_data = []
            for workload_name, metrics in results['workload_optimization'].items():
                workload_data.append({
                    'workload': workload_name,
                    'optimization_time': metrics['optimization_time_ms'],
                    'total_memory': metrics['total_memory_mb']
                })
            
            if workload_data:
                df_workload = pd.DataFrame(workload_data)
                axes[1, 0].bar(df_workload['workload'], df_workload['optimization_time'])
                axes[1, 0].set_title('Workload Optimization Time')
                axes[1, 0].set_ylabel('Time (ms)')
                
                axes[1, 1].bar(df_workload['workload'], df_workload['total_memory'])
                axes[1, 1].set_title('Optimized Memory Usage')
                axes[1, 1].set_ylabel('Memory (MB)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'buffer_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Buffer plots saved to {self.output_dir}")
    
    def save_results(self, results: Dict[str, Any]):
        """Save results to CSV."""
        csv_data = []
        
        # Memory pool results
        if 'memory_pools' in results:
            for pool_name, pool_results in results['memory_pools'].items():
                for shape_key, metrics in pool_results.items():
                    csv_data.append({
                        'benchmark_type': 'memory_pool',
                        'configuration': pool_name,
                        'shape': shape_key,
                        'allocation_time_ms': metrics['allocation_time_ms'],
                        'access_time_ms': metrics['access_time_ms'],
                        'memory_usage_mb': metrics['memory_usage_mb']
                    })
        
        # Streaming buffer results
        if 'streaming_buffers' in results:
            for capacity_key, metrics in results['streaming_buffers'].items():
                csv_data.append({
                    'benchmark_type': 'streaming_buffer',
                    'configuration': capacity_key,
                    'shape': '64x64',
                    'throughput_ops_per_sec': metrics['throughput_ops_per_sec'],
                    'avg_time_per_op_us': metrics['avg_time_per_op_us']
                })
        
        # Workload optimization results
        if 'workload_optimization' in results:
            for workload_name, metrics in results['workload_optimization'].items():
                csv_data.append({
                    'benchmark_type': 'workload_optimization',
                    'configuration': workload_name,
                    'optimization_time_ms': metrics['optimization_time_ms'],
                    'buffer_size_mb': metrics['buffer_size_mb'],
                    'num_buffers': metrics['num_buffers'],
                    'total_memory_mb': metrics['total_memory_mb']
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = self.output_dir / 'buffer_benchmark_results.csv'
            df.to_csv(csv_path, index=False)
            print(f"  📄 Results saved to {csv_path}")


def main():
    """Run buffer benchmarks."""
    benchmark = BufferBenchmark()
    results = benchmark.run_benchmarks()
    benchmark.generate_plots(results)
    benchmark.save_results(results)
    print("✅ Buffer benchmarks completed!")


if __name__ == "__main__":
    main()