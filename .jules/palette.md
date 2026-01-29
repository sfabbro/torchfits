## 2025-05-18 - Jupyter HTML Representations
**Learning:** Adding `_repr_html_` to data classes significantly improves developer experience in notebooks by providing rich, scrollable, and safe previews of complex data structures (Headers, Tables).
**Action:** Ensure all primary data objects in the library have `_repr_html_` implemented, always using `html.escape` and limiting row/column counts for performance.
