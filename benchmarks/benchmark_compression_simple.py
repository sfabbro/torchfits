2#!/usr/bin/env python3
"""
Simple compression optimization test.

Tests that small cutouts from compressed images are faster than reading full images.
"""

import time
import tempfile
import os
from pathlib import Path
import statistics

import torch
import numpy as np
from astropy.io import fits as astropy_fits
from astropy.io.fits import CompImageHDU

try:
    import torchfits
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import torchfits


def create_test_image(shape=(2000, 2000)):
    """Create a simple test image quickly."""
    np.random.seed(42)
    # Simple noise + a few bright sources
    image = np.random.normal(100, 5, shape).astype(np.float32)
    
    # Add 10 point sources
    for i in range(10):
        x, y = np.random.randint(100, shape[1]-100), np.random.randint(100, shape[0]-100)
        image[y-5:y+5, x-5:x+5] += 1000
    
    return image


def test_compression_optimization():
    """Test compression optimization for cutouts."""
    print("=== Compression Optimization Test ===")
    
    # Create test image
    print("Creating 2000x2000 test image...")
    image = create_test_image((2000, 2000))
    
    # Test RICE compression (most common in astronomy)
    compression_type = 'RICE_1'
    cutout_size = (256, 256)
    
    print(f"\\nTesting {compression_type} compression...")
    
    # Create compressed FITS file
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
        hdu = CompImageHDU(image, compression_type=compression_type)
        hdu.writeto(f.name, overwrite=True)
        filepath = f.name
    
    try:
        # Define cutout region (center)
        center_x, center_y = 1000, 1000
        half_w, half_h = 128, 128
        x1, x2 = center_x - half_w, center_x + half_w
        y1, y2 = center_y - half_h, center_y + half_h
        
        print(f"Testing cutout region: [{y1}:{y2}, {x1}:{x2}] = {cutout_size[0]}x{cutout_size[1]} pixels")
        
        # Test optimized cutout reading
        print("\\n1. Testing optimized cutout reading...")
        cutout_times = []
        cutout = None  # Initialize variable
        for i in range(5):
            start = time.perf_counter()
            # Use CFITSIO string syntax for cutout
            cutout = torchfits.read(f"{filepath}[1][{y1}:{y2},{x1}:{x2}]")
            end = time.perf_counter()
            cutout_times.append(end - start)
            print(f"   Run {i+1}: {(end-start)*1000:.1f}ms")
        
        # Test full image reading + cropping
        print("\\n2. Testing full image read + crop...")
        full_times = []
        cropped = None  # Initialize variable
        for i in range(5):
            start = time.perf_counter()
            full_image = torchfits.read(f"{filepath}[1]")
            cropped = full_image[y1:y2, x1:x2]
            end = time.perf_counter()
            full_times.append(end - start)
            print(f"   Run {i+1}: {(end-start)*1000:.1f}ms")
        
        # Calculate results
        avg_cutout_time = statistics.mean(cutout_times)
        avg_full_time = statistics.mean(full_times)
        speedup = avg_full_time / avg_cutout_time
        
        print(f"\\n=== Results ===")
        print(f"Cutout time:     {avg_cutout_time*1000:.1f} ± {statistics.stdev(cutout_times)*1000:.1f} ms")
        print(f"Full+crop time:  {avg_full_time*1000:.1f} ± {statistics.stdev(full_times)*1000:.1f} ms")
        print(f"Speedup:         {speedup:.2f}x")
        
        # Validate results
        print(f"\\n=== Validation ===")
        if cutout is not None and cropped is not None:
            print(f"Cutout shape:    {cutout.shape}")
            print(f"Cropped shape:   {cropped.shape}")
            print(f"Data matches:    {torch.allclose(cutout, cropped, rtol=1e-4)}")
        
        # Performance assessment
        print(f"\\n=== Assessment ===")
        cutout_fraction = (cutout_size[0] * cutout_size[1]) / (image.shape[0] * image.shape[1])
        print(f"Cutout fraction: {cutout_fraction:.4f} ({cutout_fraction*100:.2f}% of full image)")
        
        if speedup > 2.0:
            print("✅ EXCELLENT: Compression optimization is working very well!")
            print("   Small cutouts are significantly faster than full decompression.")
        elif speedup > 1.5:
            print("✅ GOOD: Compression optimization is working!")
            print("   Noticeable performance benefit for small cutouts.")
        elif speedup > 1.1:
            print("⚠️  PARTIAL: Some optimization benefit detected")
            print("   Minor performance improvement.")
        else:
            print("❌ ISSUE: No significant optimization benefit")
            print("   Cutouts are not faster than full image reads.")
            
        # Expected theoretical speedup
        theoretical_speedup = 1.0 / cutout_fraction
        efficiency = speedup / theoretical_speedup
        print(f"\\nTheoretical max speedup: {theoretical_speedup:.1f}x")
        print(f"Optimization efficiency:  {efficiency*100:.1f}%")
        
        return speedup > 1.1
        
    except Exception as e:
        print(f"Error during test: {e}")
        return False
        
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


def main():
    """Run compression optimization test."""
    print("Compression Optimization Test")
    print("=" * 50)
    
    success = test_compression_optimization()
    
    print("\\n" + "=" * 50)
    if success:
        print("✅ COMPRESSION OPTIMIZATION WORKING!")
        print("   Task #3 from OPTIMIZE.md has been successfully implemented.")
        print("   Tiled decompression provides faster cutout reading.")
    else:
        print("❌ Compression optimization needs investigation.")
    
    print("\\nSummary: The compression optimization uses CFITSIO's built-in")
    print("tile-aware decompression to only decompress the necessary tiles") 
    print("for small cutouts, rather than decompressing the entire image.")


if __name__ == "__main__":
    main()