## Bolt's Journal

## 2024-05-14 - Fast EXTNAME lookup in HDUList
**Learning:** `HDUList.__getitem__` iterated over all HDUs to find a matching EXTNAME, which is O(N) and slow for files with many extensions, or when accessed repeatedly.
**Action:** Adding a cached dictionary mapping `EXTNAME` to HDU index makes lookup O(1) and ~40x faster for repeated lookups with many HDUs.
