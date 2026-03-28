
## 2025-03-05 - FITS Header Parsing Optimization
**Learning:** In a performance-sensitive parser loop like reading FITS headers (which can have thousands of 80-character cards), Python function call overhead and class attribute lookups (e.g., `cls._STRING_KEYWORDS`) can be a significant bottleneck. Inlining the parsing logic and aliasing class attributes to local variables provided a ~20% speedup.
**Action:** When working on tight loops or parsing functions, look for opportunities to inline small helper functions and locally alias frequently accessed class attributes to reduce lookup overhead.
