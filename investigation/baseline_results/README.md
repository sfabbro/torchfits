# Torchfits Performance Baseline (Phase 0)

## Test Setup
- Platform: macOS ARM64 (Apple Silicon)
- Python: 3.13  
- Test: 1000x1000 float32 image
- Caching: DISABLED for accurate I/O measurement

## Baseline Performance (November 27, 2024)

### Without Caching (Real I/O):
- torchfits: 1.93ms ± 2.19ms  ← **CURRENT (using astropy backend)**
- astropy:   0.23ms ± 0.01ms  ← **Direct astropy**
- fitsio:    0.58ms ± 0.14ms

### With Caching (After warm-up):
- torchfits: 0.006-0.010ms (cache hits)
- Competitors: 0.2-10ms (no cache)

## Critical Findings

1. **torchfits is 8x SLOWER than astropy** when cache is disabled
   - Root cause: Python read() function uses astropy internally (src/torchfits/__init__.py lines 130-268)
   - The C++ backend exists but isn't being used!

2. **torchfits has excellent caching**
   - Python-side cache makes repeated reads 200-1000x faster
   - But hides the underlying I/O performance problem

3. **Phase 1 is CRITICAL**
   - Must replace astropy fallback with C++ backend
   - Expected improvement: 2-5x faster than astropy
   - Target: <0.1ms for 1000x1000 images

## Next Steps

Phase 1: Fix Python-to-C++ bridge to use the compiled C++ backend instead of astropy.
Expected result: 0.05-0.12ms (2-5x faster than astropy)
