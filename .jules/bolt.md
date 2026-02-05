## 2024-05-23 - [Function Multi-Versioning for Auto-Vectorization]
**Learning:** Manual SIMD intrinsics (AVX2) were significantly SLOWER (2.9 GB/s vs 3.6 GB/s) than GCC's auto-vectorization of the scalar loop when targeting AVX2. This is because modern compilers are extremely good at vectorizing simple loops (like linear scaling `val * scale + zero`). The key was enabling AVX2 instruction set for the function.
**Action:** Use Function Multi-Versioning (`__attribute__((target_clones("avx2", "default")))`) to enable AVX2 auto-vectorization for specific hot functions without raising the global architecture baseline or maintaining complex manual intrinsic paths.

## 2025-02-14 - [Python String Parsing Optimization]
**Learning:** Python loops for character-by-character string parsing are extremely slow. For FITS header parsing, replacing manual loops with `str.find` (for simple cases) and compiled Regex (for complex quoted strings) resulted in 3x-12x speedup.
**Action:** Always prefer `str` methods or Regex over manual iteration for string processing in hot paths, especially for parsing.
