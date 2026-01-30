# Bolt's Journal

## 2025-05-23 - FITS Header Parsing Optimization
**Learning:** Python string parsing of fixed-width records (FITS cards) can be significantly optimized by avoiding character-level loops. Using regex for quoted strings and fast-path logic for unquoted values (numbers/booleans) yields ~20% speedup.
**Action:** When parsing well-defined formats in Python, prefer regex or string methods (split/find) over manual iteration, especially for hot paths.
