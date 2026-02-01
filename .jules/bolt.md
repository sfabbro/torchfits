## 2024-05-23 - [Function Multi-Versioning for Auto-Vectorization]
**Learning:** Manual SIMD intrinsics (AVX2) were significantly SLOWER (2.9 GB/s vs 3.6 GB/s) than GCC's auto-vectorization of the scalar loop when targeting AVX2. This is because modern compilers are extremely good at vectorizing simple loops (like linear scaling `val * scale + zero`). The key was enabling AVX2 instruction set for the function.
**Action:** Use Function Multi-Versioning (`__attribute__((target_clones("avx2", "default")))`) to enable AVX2 auto-vectorization for specific hot functions without raising the global architecture baseline or maintaining complex manual intrinsic paths.

## 2025-02-21 - [Unused and Unexposed C++ Cache Implementation]
**Learning:** Found a sophisticated C++ cache implementation (`UnifiedCache`) that was completely unused because its Python bindings were missing and WCS dependency issues prevented compilation in some environments. Sometimes "optimizing" means just enabling the code that's already there but broken/hidden. Also, standard `std::list::erase` + `push_front` for LRU is suboptimal compared to `splice`, which avoids allocation.
**Action:** When porting or maintaining C++/Python extensions, verify that "optional" C++ components are actually exposed to Python and tested. Use `std::list::splice` for O(1) node movement in LRU implementations.
