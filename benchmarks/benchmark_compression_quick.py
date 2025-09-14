#!/usr/bin/env python3
"""
Quick compression optimization benchmark.

Tests the key compression optimization: small cutouts from compressed images
should be faster than reading the full image and cropping.
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
    print("=== Quick Compression Optimization Test ===")
    
    # Create test image
    print("Creating test image...")
    image = create_test_image((2000, 2000))
    
    # Test different compression types
    compression_types = ['RICE_1', 'GZIP_1']
    cutout_size = (256, 256)
    
    results = []
    
    for compression_type in compression_types:
        print(f"\nTesting {compression_type} compression...")
        
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
            
            # Test optimized cutout reading
            print("  Testing optimized cutout reading...")
            cutout_times = []
            for _ in range(3):
                start = time.perf_counter()
                # Use slice notation for cutout reading
                cutout = torchfits.read(f"{filepath}[1][{y1}:{y2},{x1}:{x2}]")
                end = time.perf_counter()
                cutout_times.append(end - start)
            
            # Test full image reading + cropping
            print("  Testing full image + crop...")
            full_times = []
            for _ in range(3):
                start = time.perf_counter()
                with torchfits.open(filepath) as f:
                    full_image = torchfits.read(f"{filepath}[1]")
                    cropped = full_image[y1:y2, x1:x2]
                end = time.perf_counter()
                full_times.append(end - start)
            
            # Calculate results
            avg_cutout_time = statistics.mean(cutout_times)
            avg_full_time = statistics.mean(full_times)
            speedup = avg_full_time / avg_cutout_time
            
            results.append({
                'compression': compression_type,
                'cutout_time': avg_cutout_time,
                'full_time': avg_full_time,
                'speedup': speedup
            })
            
            print(f"  Results:")
            print(f"    Cutout time: {avg_cutout_time:.4f}s")
            print(f"    Full+crop time: {avg_full_time:.4f}s")
            print(f"    Speedup: {speedup:.2f}x")
            
            # Test compression metadata
            with torchfits.open(filepath) as f:
                header = f[1].get_header()
                compressed = header.get('_TORCHFITS_COMPRESSED', 'False')
                comp_type = header.get('_TORCHFITS_COMPRESSION_TYPE', '0')
                tile_dim1 = header.get('_TORCHFITS_TILE_DIM1', '0')
                tile_dim2 = header.get('_TORCHFITS_TILE_DIM2', '0')
                
                print(f"  Compression metadata:")
                print(f"    Compressed: {compressed}")
                print(f"    Type ID: {comp_type}")
                print(f"    Tile dims: {tile_dim1}x{tile_dim2}")
            
        except Exception as e:
            print(f"  Error: {e}")
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    # Summary
    print("\n=== Summary ===")
    if results:
        avg_speedup = statistics.mean([r['speedup'] for r in results])
        print(f"Average speedup: {avg_speedup:.2f}x")
        
        if avg_speedup > 1.5:
            print("✅ SUCCESS: Compression optimization is working!")
            print("   Small cutouts from compressed images are significantly faster")
            print("   than reading full images and cropping.")
        elif avg_speedup > 1.1:
            print("⚠️  PARTIAL SUCCESS: Some optimization benefit detected")
        else:
            print("❌ ISSUE: No significant optimization benefit")
            
        # Detailed results
        for r in results:
            print(f"{r['compression']}: {r['speedup']:.2f}x speedup")
    else:
        print("❌ No successful tests completed")


def test_compression_detection():
    """Test compression detection functionality."""
    print("\n=== Compression Detection Test ===")
    
    # Create uncompressed file
    image = create_test_image((500, 500))
    
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
        hdu = astropy_fits.PrimaryHDU(image)
        hdu.writeto(f.name, overwrite=True)
        uncompressed_file = f.name
    
    # Create compressed file
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
        hdu = CompImageHDU(image, compression_type='RICE_1')
        hdu.writeto(f.name, overwrite=True)
        compressed_file = f.name
    
    try:
        # Test uncompressed detection
        print("Testing uncompressed file...")
        with torchfits.open(uncompressed_file) as f:
            header = f[0].get_header()
            compressed = header.get('_TORCHFITS_COMPRESSED', 'Unknown')
            print(f"  Detected as compressed: {compressed}")
        
        # Test compressed detection
        print("Testing compressed file...")
        with torchfits.open(compressed_file) as f:
            header = f[1].get_header()  # Compressed image is in HDU 1
            compressed = header.get('_TORCHFITS_COMPRESSED', 'Unknown')
            comp_type = header.get('_TORCHFITS_COMPRESSION_TYPE', '0')
            print(f"  Detected as compressed: {compressed}")
            print(f"  Compression type ID: {comp_type}")
            
    finally:
        os.unlink(uncompressed_file)
        os.unlink(compressed_file)


def main():
    """Run quick compression tests."""
    print("Quick Compression Optimization Benchmark")
    print("=" * 50)
    
    test_compression_detection()
    test_compression_optimization()
    
    print("\n" + "=" * 50)
    print("Compression optimization testing complete!")


if __name__ == "__main__":
    main()