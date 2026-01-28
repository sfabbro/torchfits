## 2024-05-23 - [Function Multi-Versioning for Auto-Vectorization]
**Learning:** Manual SIMD intrinsics (AVX2) were significantly SLOWER (2.9 GB/s vs 3.6 GB/s) than GCC's auto-vectorization of the scalar loop when targeting AVX2. This is because modern compilers are extremely good at vectorizing simple loops (like linear scaling `val * scale + zero`). The key was enabling AVX2 instruction set for the function.
**Action:** Use Function Multi-Versioning (`__attribute__((target_clones("avx2", "default")))`) to enable AVX2 auto-vectorization for specific hot functions without raising the global architecture baseline or maintaining complex manual intrinsic paths.

## 2024-05-24 - [Python Regex vs Manual Parsing]
**Learning:** For parsing mixed FITS value types (integers, floats, strings, booleans), a compiled Regex (e.g., `re.compile(r"^[+-]?\d+$")`) was consistently FASTER (8.8ms vs 11.3ms) than manual Python string checks (e.g., `s.isdigit()`) coupled with `try/except` blocks. The overhead of Python logic/function calls outweighs the regex engine's setup cost when many checks fail (e.g. checking float string as int).
**Action:** Prefer compiled Regex for validation/parsing of mixed inputs in tight loops over pure Python logic unless the input is highly uniform.
