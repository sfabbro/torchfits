#!/usr/bin/env python3
"""
Enhanced benchmark runner with logging, resume capability, and robust error handling.
"""

import sys
import os
import time
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

class BenchmarkRunner:
    """Enhanced benchmark runner with logging and resume capability."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("benchmark_results_new")
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.log_file = self.output_dir / "benchmark_log.txt"
        self.setup_logging()
        
        # State file for resume capability
        self.state_file = self.output_dir / "benchmark_state.json"
        self.state = self.load_state()
        
        # Available benchmarks
        self.benchmarks = {
            'basic': {
                'script': 'benchmark_basic.py',
                'description': 'Basic FITS I/O performance tests',
                'estimated_time': 120,  # seconds
                'priority': 1
            },
            'core': {
                'script': 'benchmark_core.py', 
                'description': 'Core functionality benchmarks',
                'estimated_time': 180,
                'priority': 1
            },
            'table': {
                'script': 'benchmark_table.py',
                'description': 'Table I/O performance tests',
                'estimated_time': 240,
                'priority': 2
            },
            'table_comprehensive': {
                'script': 'benchmark_table_comprehensive.py',
                'description': 'Comprehensive table benchmarks',
                'estimated_time': 300,
                'priority': 2
            },
            'ml': {
                'script': 'benchmark_ml.py',
                'description': 'Machine learning pipeline benchmarks',
                'estimated_time': 180,
                'priority': 2
            },
            'transforms': {
                'script': 'benchmark_transforms.py',
                'description': 'Data transformation benchmarks',
                'estimated_time': 150,
                'priority': 3
            },
            'buffer': {
                'script': 'benchmark_buffer.py',
                'description': 'Buffer management benchmarks',
                'estimated_time': 120,
                'priority': 3
            },
            'cache': {
                'script': 'benchmark_cache.py',
                'description': 'Cache system benchmarks',
                'estimated_time': 180,
                'priority': 3
            },
            'compression': {
                'script': 'benchmark_compression_final.py',
                'description': 'Compression performance tests',
                'estimated_time': 360,
                'priority': 2
            },
            'cpp_backend': {
                'script': 'benchmark_cpp_backend.py',
                'description': 'C++ backend performance tests',
                'estimated_time': 240,
                'priority': 2
            },
            'gpu_memory': {
                'script': 'benchmark_gpu_memory.py',
                'description': 'GPU memory management tests',
                'estimated_time': 180,
                'priority': 3
            },
            'mmap': {
                'script': 'benchmark_mmap.py',
                'description': 'Memory mapping benchmarks',
                'estimated_time': 150,
                'priority': 3
            },
            'header_parsing': {
                'script': 'benchmark_header_parsing.py',
                'description': 'Header parsing performance tests',
                'estimated_time': 120,
                'priority': 3
            },
            'zero_copy': {
                'script': 'benchmark_zero_copy_optimizations.py',
                'description': 'Zero-copy optimization tests',
                'estimated_time': 180,
                'priority': 3
            },
            'variance': {
                'script': 'benchmark_variance_analysis.py',
                'description': 'Performance variance analysis',
                'estimated_time': 300,
                'priority': 4
            },
            'regression': {
                'script': 'benchmark_performance_regression.py',
                'description': 'Performance regression tests',
                'estimated_time': 240,
                'priority': 4
            },
            'exhaustive': {
                'script': 'benchmark_all.py',
                'description': 'Exhaustive benchmark suite (all scenarios)',
                'estimated_time': 1800,  # 30 minutes
                'priority': 5
            }
        }
        
    def setup_logging(self):
        """Setup logging to both file and console."""
        # Create logger
        self.logger = logging.getLogger('benchmark_runner')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def load_state(self) -> Dict:
        """Load benchmark state for resume capability."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load state file: {e}")
        
        return {
            'completed': [],
            'failed': [],
            'started_at': None,
            'last_update': None
        }
    
    def save_state(self):
        """Save current benchmark state."""
        self.state['last_update'] = datetime.now().isoformat()
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save state: {e}")
    
    def run_benchmark(self, name: str, benchmark_info: Dict) -> Tuple[bool, str, str]:
        """Run a single benchmark and capture output."""
        script_path = Path(__file__).parent / benchmark_info['script']
        
        if not script_path.exists():
            return False, f"Script not found: {script_path}", ""
        
        self.logger.info(f"Starting benchmark: {name} ({benchmark_info['description']})")
        self.logger.info(f"Estimated time: {benchmark_info['estimated_time']} seconds")
        
        start_time = time.time()
        
        try:
            # Run benchmark with timeout
            timeout = benchmark_info['estimated_time'] * 3  # 3x estimated time as timeout
            
            # Add src to PYTHONPATH
            env = os.environ.copy()
            src_path = str(Path(__file__).parent.parent / 'src')
            env['PYTHONPATH'] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}"

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=script_path.parent,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.info(f"✅ {name} completed successfully in {elapsed_time:.1f}s")
                return True, result.stdout, result.stderr
            else:
                self.logger.error(f"❌ {name} failed with return code {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr}")
                return False, result.stdout, result.stderr
                
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            self.logger.error(f"⏰ {name} timed out after {elapsed_time:.1f}s")
            return False, "", f"Benchmark timed out after {elapsed_time:.1f}s"
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"💥 {name} crashed: {e}")
            return False, "", str(e)
    
    def run_benchmarks(self, benchmark_names: List[str], resume: bool = True) -> Dict:
        """Run specified benchmarks with resume capability."""
        if not self.state['started_at']:
            self.state['started_at'] = datetime.now().isoformat()
        
        results = {
            'started_at': self.state['started_at'],
            'completed': [],
            'failed': [],
            'skipped': [],
            'total_time': 0
        }
        
        # Filter benchmarks to run
        to_run = []
        for name in benchmark_names:
            if name not in self.benchmarks:
                self.logger.warning(f"Unknown benchmark: {name}")
                continue
                
            if resume and name in self.state['completed']:
                self.logger.info(f"⏭️  Skipping {name} (already completed)")
                results['skipped'].append(name)
                continue
                
            if resume and name in self.state['failed']:
                self.logger.info(f"🔄 Retrying {name} (previously failed)")
            
            to_run.append(name)
        
        if not to_run:
            self.logger.info("No benchmarks to run")
            return results
        
        # Estimate total time
        total_estimated = sum(self.benchmarks[name]['estimated_time'] for name in to_run)
        self.logger.info(f"Running {len(to_run)} benchmarks (estimated time: {total_estimated//60}m {total_estimated%60}s)")
        
        overall_start = time.time()
        
        for i, name in enumerate(to_run, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Benchmark {i}/{len(to_run)}: {name}")
            self.logger.info(f"{'='*60}")
            
            benchmark_start = time.time()
            success, stdout, stderr = self.run_benchmark(name, self.benchmarks[name])
            benchmark_time = time.time() - benchmark_start
            
            # Save outputs
            if stdout:
                stdout_file = self.output_dir / f"{name}_stdout.txt"
                with open(stdout_file, 'w') as f:
                    f.write(stdout)
            
            if stderr:
                stderr_file = self.output_dir / f"{name}_stderr.txt"
                with open(stderr_file, 'w') as f:
                    f.write(stderr)
            
            # Update state and results
            if success:
                results['completed'].append({
                    'name': name,
                    'time': benchmark_time,
                    'description': self.benchmarks[name]['description']
                })
                self.state['completed'].append(name)
                if name in self.state['failed']:
                    self.state['failed'].remove(name)
            else:
                results['failed'].append({
                    'name': name,
                    'time': benchmark_time,
                    'description': self.benchmarks[name]['description'],
                    'error': stderr or "Unknown error"
                })
                if name not in self.state['failed']:
                    self.state['failed'].append(name)
            
            self.save_state()
            
            # Progress update
            elapsed_total = time.time() - overall_start
            remaining = len(to_run) - i
            if i > 0:
                avg_time = elapsed_total / i
                eta = avg_time * remaining
                self.logger.info(f"Progress: {i}/{len(to_run)} completed, ETA: {eta//60:.0f}m {eta%60:.0f}s")
        
        results['total_time'] = time.time() - overall_start
        results['finished_at'] = datetime.now().isoformat()
        
        return results
    
    def generate_summary(self, results: Dict):
        """Generate a summary report."""
        summary_file = self.output_dir / "benchmark_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write("# Benchmark Summary Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if results['started_at']:
                f.write(f"Started: {results['started_at']}\n")
            if 'finished_at' in results:
                f.write(f"Finished: {results['finished_at']}\n")
            f.write(f"Total time: {results['total_time']//60:.0f}m {results['total_time']%60:.0f}s\n\n")
            
            # Completed benchmarks
            if results['completed']:
                f.write(f"## ✅ Completed Benchmarks ({len(results['completed'])})\n\n")
                for bench in results['completed']:
                    f.write(f"- **{bench['name']}**: {bench['description']} ({bench['time']:.1f}s)\n")
                f.write("\n")
            
            # Failed benchmarks
            if results['failed']:
                f.write(f"## ❌ Failed Benchmarks ({len(results['failed'])})\n\n")
                for bench in results['failed']:
                    f.write(f"- **{bench['name']}**: {bench['description']} ({bench['time']:.1f}s)\n")
                    f.write(f"  Error: {bench['error'][:100]}...\n")
                f.write("\n")
            
            # Skipped benchmarks
            if results['skipped']:
                f.write(f"## ⏭️ Skipped Benchmarks ({len(results['skipped'])})\n\n")
                for name in results['skipped']:
                    f.write(f"- **{name}**: {self.benchmarks[name]['description']}\n")
                f.write("\n")
            
            # Performance summary
            if results['completed']:
                total_bench_time = sum(b['time'] for b in results['completed'])
                f.write("## Performance Summary\n\n")
                f.write(f"- Benchmarks completed: {len(results['completed'])}\n")
                f.write(f"- Total benchmark time: {total_bench_time//60:.0f}m {total_bench_time%60:.0f}s\n")
                f.write(f"- Average time per benchmark: {total_bench_time/len(results['completed']):.1f}s\n")
                
                fastest = min(results['completed'], key=lambda x: x['time'])
                slowest = max(results['completed'], key=lambda x: x['time'])
                f.write(f"- Fastest: {fastest['name']} ({fastest['time']:.1f}s)\n")
                f.write(f"- Slowest: {slowest['name']} ({slowest['time']:.1f}s)\n")
                f.write("\n")
            
            # Next steps
            f.write("## Next Steps\n\n")
            if results['failed']:
                f.write("1. Review failed benchmark logs for specific issues\n")
                f.write("2. Fix any identified problems and re-run failed benchmarks\n")
            f.write("3. Review individual benchmark outputs for performance insights\n")
            f.write("4. Compare results with previous runs if available\n")
        
        self.logger.info(f"Summary report saved to: {summary_file}")
    
    def list_benchmarks(self):
        """List all available benchmarks."""
        print("\nAvailable benchmarks:")
        print("=" * 60)
        
        by_priority = {}
        for name, info in self.benchmarks.items():
            priority = info['priority']
            if priority not in by_priority:
                by_priority[priority] = []
            by_priority[priority].append((name, info))
        
        for priority in sorted(by_priority.keys()):
            print(f"\nPriority {priority}:")
            for name, info in sorted(by_priority[priority]):
                status = ""
                if name in self.state['completed']:
                    status = " ✅"
                elif name in self.state['failed']:
                    status = " ❌"
                
                print(f"  {name:20s} - {info['description']} ({info['estimated_time']//60}m {info['estimated_time']%60}s){status}")
    
    def reset_state(self):
        """Reset benchmark state."""
        self.state = {
            'completed': [],
            'failed': [],
            'started_at': None,
            'last_update': None
        }
        self.save_state()
        self.logger.info("Benchmark state reset")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Enhanced benchmark runner')
    parser.add_argument('--output-dir', type=Path, default=Path('benchmark_results_new'),
                        help='Output directory for results')
    parser.add_argument('--list', action='store_true',
                        help='List available benchmarks')
    parser.add_argument('--reset', action='store_true',
                        help='Reset benchmark state')
    parser.add_argument('--no-resume', action='store_true',
                        help='Do not resume from previous state')
    parser.add_argument('--priority', type=int, choices=[1, 2, 3, 4, 5],
                        help='Run benchmarks up to specified priority level')
    parser.add_argument('benchmarks', nargs='*',
                        help='Specific benchmarks to run (default: priority 1-2)')
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(output_dir=args.output_dir)
    
    if args.list:
        runner.list_benchmarks()
        return
    
    if args.reset:
        runner.reset_state()
        return
    
    # Determine which benchmarks to run
    if args.benchmarks:
        benchmark_names = args.benchmarks
    elif args.priority:
        benchmark_names = [
            name for name, info in runner.benchmarks.items()
            if info['priority'] <= args.priority
        ]
    else:
        # Default: priority 1-2 (essential benchmarks)
        benchmark_names = [
            name for name, info in runner.benchmarks.items()
            if info['priority'] <= 2
        ]
    
    # Sort by priority
    benchmark_names.sort(key=lambda x: (runner.benchmarks[x]['priority'], x))
    
    runner.logger.info("Starting benchmark runner")
    runner.logger.info(f"Output directory: {runner.output_dir}")
    
    try:
        results = runner.run_benchmarks(benchmark_names, resume=not args.no_resume)
        runner.generate_summary(results)
        
        # Final summary
        print("\n" + "=" * 60)
        print("BENCHMARK RUNNER COMPLETED")
        print("=" * 60)
        print(f"Completed: {len(results['completed'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Skipped: {len(results['skipped'])}")
        print(f"Total time: {results['total_time']//60:.0f}m {results['total_time']%60:.0f}s")
        print(f"Results saved to: {runner.output_dir}")
        
        if results['failed']:
            print("\n⚠️  Some benchmarks failed. Check the logs for details.")
            sys.exit(1)
        else:
            print("\n✅ All benchmarks completed successfully!")
            
    except KeyboardInterrupt:
        runner.logger.info("Benchmark runner interrupted by user")
        print("\n⏹️  Benchmark runner interrupted. State saved for resume.")
        sys.exit(130)
    except Exception as e:
        runner.logger.error(f"Benchmark runner failed: {e}")
        print(f"\n💥 Benchmark runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()