"""
Statistical Analysis of Performance Variance vs Overhead

This experiment specifically tests whether the performance discrepancy 
in small files is due to measurement variance or actual overhead.
"""

import time
import sys
from pathlib import Path
import numpy as np
import torch
import tempfile
import os
import statistics
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torchfits
except ImportError as e:
    print(f"⚠️  torchfits import failed: {e}")
    torchfits = None

try:
    from astropy.io import fits as astropy_fits
except ImportError:
    astropy_fits = None

try:
    import fitsio
except ImportError:
    fitsio = None


def create_test_file(shape, dtype):
    """Create a test FITS file."""
    data = np.random.randn(*shape).astype(dtype)
    
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
        if astropy_fits:
            astropy_fits.writeto(f.name, data, overwrite=True)
            return f.name, data
    return None, None


def precise_benchmark(func, num_runs=100, warmup_runs=10):
    """Perform precise benchmarking with statistical analysis."""
    # Warmup runs to stabilize performance
    for _ in range(warmup_runs):
        try:
            func()
        except:
            pass
    
    # Actual timing runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times.append(end - start)
    
    return np.array(times)


def analyze_distribution(times, label):
    """Analyze the statistical distribution of timing measurements."""
    mean_time = np.mean(times)
    std_time = np.std(times)
    median_time = np.median(times)
    min_time = np.min(times)
    max_time = np.max(times)
    cv = std_time / mean_time  # Coefficient of variation
    
    # Remove outliers (beyond 2 standard deviations)
    clean_times = times[np.abs(times - mean_time) <= 2 * std_time]
    clean_mean = np.mean(clean_times)
    clean_std = np.std(clean_times)
    
    print(f"  {label}:")
    print(f"    Raw:   {mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms (CV: {cv:.3f})")
    print(f"    Clean: {clean_mean*1000:.3f}ms ± {clean_std*1000:.3f}ms ({len(clean_times)}/{len(times)} samples)")
    print(f"    Range: {min_time*1000:.3f}ms - {max_time*1000:.3f}ms")
    print(f"    Median: {median_time*1000:.3f}ms")
    
    return {
        'mean': mean_time,
        'std': std_time,
        'clean_mean': clean_mean,
        'clean_std': clean_std,
        'median': median_time,
        'cv': cv,
        'n_clean': len(clean_times),
        'n_total': len(times)
    }


def statistical_comparison(times1, times2, label1, label2):
    """Perform statistical tests to compare two timing distributions."""
    # Remove outliers from both
    mean1, std1 = np.mean(times1), np.std(times1)
    mean2, std2 = np.mean(times2), np.std(times2)
    
    clean1 = times1[np.abs(times1 - mean1) <= 2 * std1]
    clean2 = times2[np.abs(times2 - mean2) <= 2 * std2]
    
    # Mann-Whitney U test (non-parametric)
    statistic, p_value = stats.mannwhitneyu(clean1, clean2, alternative='two-sided')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(clean1)-1)*np.var(clean1) + (len(clean2)-1)*np.var(clean2)) / (len(clean1)+len(clean2)-2))
    cohens_d = (np.mean(clean1) - np.mean(clean2)) / pooled_std
    
    # Confidence interval for difference in means
    diff_mean = np.mean(clean1) - np.mean(clean2)
    diff_std = np.sqrt(np.var(clean1)/len(clean1) + np.var(clean2)/len(clean2))
    ci_95 = 1.96 * diff_std
    
    print(f"\n📊 Statistical Comparison: {label1} vs {label2}")
    print(f"  Difference in means: {diff_mean*1000:.3f}ms ± {ci_95*1000:.3f}ms (95% CI)")
    print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
    print(f"  Mann-Whitney U p-value: {p_value:.6f}")
    
    # Interpretation
    if p_value < 0.001:
        significance = "highly significant"
    elif p_value < 0.01:
        significance = "significant"
    elif p_value < 0.05:
        significance = "marginally significant"
    else:
        significance = "not significant"
    
    if abs(cohens_d) > 0.8:
        effect = "large"
    elif abs(cohens_d) > 0.5:
        effect = "medium"
    elif abs(cohens_d) > 0.2:
        effect = "small"
    else:
        effect = "negligible"
    
    print(f"  Interpretation: {significance} difference ({effect} effect)")
    
    return {
        'p_value': p_value,
        'cohens_d': cohens_d,
        'diff_mean': diff_mean,
        'diff_ci': ci_95,
        'significance': significance,
        'effect': effect
    }


def test_measurement_consistency():
    """Test measurement consistency by running the same operation multiple times."""
    print("🔬 Testing Measurement Consistency")
    print("=" * 50)
    
    # Create a very small test file
    shape = (50, 50)  # Very small to maximize relative overhead
    filepath, _ = create_test_file(shape, np.float32)
    
    if not filepath or not torchfits:
        print("❌ Failed to create test file or import torchfits")
        return
    
    try:
        # Test the same operation multiple times with large sample sizes
        print(f"Testing {shape} float32 with 200 runs each...\n")
        
        # Torchfits run 1
        times1 = precise_benchmark(lambda: torchfits.read(filepath), num_runs=200)
        stats1 = analyze_distribution(times1, "torchfits run 1")
        
        # Torchfits run 2 (same operation)
        times2 = precise_benchmark(lambda: torchfits.read(filepath), num_runs=200)
        stats2 = analyze_distribution(times2, "torchfits run 2")
        
        # Compare the two identical operations
        comparison = statistical_comparison(times1, times2, "torchfits run 1", "torchfits run 2")
        
        # If there's significant difference between identical operations, 
        # it indicates high measurement variance
        if comparison['p_value'] < 0.05:
            print("⚠️  HIGH VARIANCE: Identical operations show significant differences")
            print("    This suggests measurement noise dominates for small files")
        else:
            print("✅ LOW VARIANCE: Identical operations are statistically equivalent")
            print("    Measurements are reliable for detecting real differences")
        
        # Now compare with fitsio if available
        if fitsio:
            print(f"\n" + "="*50)
            print("Comparing torchfits vs fitsio with reliable statistics...\n")
            
            def fitsio_read():
                array = fitsio.read(filepath)
                return torch.from_numpy(array)
            
            fitsio_times = precise_benchmark(fitsio_read, num_runs=200)
            fitsio_stats = analyze_distribution(fitsio_times, "fitsio")
            
            # Statistical comparison
            torch_vs_fitsio = statistical_comparison(times1, fitsio_times, "torchfits", "fitsio")
            
            # Calculate speedup with confidence intervals
            speedup = fitsio_stats['clean_mean'] / stats1['clean_mean']
            speedup_error = speedup * np.sqrt(
                (fitsio_stats['clean_std']/fitsio_stats['clean_mean'])**2 + 
                (stats1['clean_std']/stats1['clean_mean'])**2
            )
            
            print(f"\n🎯 FINAL RESULT:")
            print(f"  Speedup: {speedup:.3f}x ± {speedup_error:.3f}x")
            
            if torch_vs_fitsio['p_value'] < 0.01 and torch_vs_fitsio['effect'] != 'negligible':
                if speedup > 1:
                    print(f"  ✅ torchfits is reliably faster than fitsio")
                else:
                    print(f"  ⚠️  torchfits is reliably slower than fitsio")
                print(f"     (statistical evidence: {torch_vs_fitsio['significance']} with {torch_vs_fitsio['effect']} effect)")
            else:
                print(f"  🤷 Performance difference is within measurement variance")
                print(f"     (too close to call reliably)")
        
    finally:
        os.unlink(filepath)


def test_multiple_file_sizes():
    """Test variance vs overhead across multiple file sizes."""
    print("\n" + "🔬 Multi-Size Variance Analysis")
    print("=" * 50)
    
    test_sizes = [
        ((25, 25), "tiny"),
        ((100, 100), "small"), 
        ((500, 500), "medium"),
        ((1000, 1000), "large")
    ]
    
    results = []
    
    for shape, size_label in test_sizes:
        print(f"\n📊 Testing {size_label} files {shape}")
        
        filepath, _ = create_test_file(shape, np.float32)
        if not filepath:
            continue
            
        try:
            # High precision measurements
            torch_times = precise_benchmark(lambda: torchfits.read(filepath), num_runs=100)
            torch_stats = analyze_distribution(torch_times, f"torchfits {size_label}")
            
            if fitsio:
                fitsio_times = precise_benchmark(
                    lambda: torch.from_numpy(fitsio.read(filepath)), 
                    num_runs=100
                )
                fitsio_stats = analyze_distribution(fitsio_times, f"fitsio {size_label}")
                
                comparison = statistical_comparison(torch_times, fitsio_times, "torchfits", "fitsio")
                
                speedup = fitsio_stats['clean_mean'] / torch_stats['clean_mean'] 
                
                results.append({
                    'size': size_label,
                    'shape': shape,
                    'speedup': speedup,
                    'p_value': comparison['p_value'],
                    'effect': comparison['effect'],
                    'torch_cv': torch_stats['cv'],
                    'fitsio_cv': fitsio_stats['cv']
                })
                
                print(f"  Speedup: {speedup:.3f}x ({comparison['significance']})")
        
        finally:
            os.unlink(filepath)
    
    # Analyze trends
    if results:
        print(f"\n📈 TREND ANALYSIS:")
        print(f"{'Size':<8} {'Speedup':<8} {'P-value':<10} {'Effect':<12} {'TorchCV':<8} {'FitsCV':<8}")
        print("-" * 60)
        
        for r in results:
            print(f"{r['size']:<8} {r['speedup']:<8.3f} {r['p_value']:<10.6f} {r['effect']:<12} {r['torch_cv']:<8.3f} {r['fitsio_cv']:<8.3f}")
        
        # Check if variance increases for smaller files
        tiny_cv = next((r['torch_cv'] for r in results if r['size'] == 'tiny'), None)
        large_cv = next((r['torch_cv'] for r in results if r['size'] == 'large'), None)
        
        if tiny_cv and large_cv:
            if tiny_cv > large_cv * 2:
                print(f"\n⚠️  VARIANCE INCREASES for smaller files (CV: {tiny_cv:.3f} vs {large_cv:.3f})")
                print("    This suggests measurement noise affects small file results")
            else:
                print(f"\n✅ VARIANCE is consistent across file sizes")
                print("    Performance differences are likely real, not measurement artifacts")


def main():
    """Run comprehensive variance vs overhead analysis."""
    print("🧪 Performance Variance vs Overhead Analysis")
    print("=" * 60)
    
    if not torchfits:
        print("❌ torchfits not available")
        return
    
    if not fitsio:
        print("❌ fitsio not available - cannot compare")
        return
    
    # Test measurement consistency
    test_measurement_consistency()
    
    # Test across multiple file sizes
    test_multiple_file_sizes()
    
    print("\n" + "=" * 60)
    print("Analysis complete!")


if __name__ == "__main__":
    main()