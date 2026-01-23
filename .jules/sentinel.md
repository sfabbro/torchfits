## 2026-01-23 - [FITS Header and Column Sanitization]
**Vulnerability:** FITS headers (keys, comments) and table column names were not sanitized when read from files.
**Learning:** External data formats (like FITS) often have loose specifications or implementations that allow arbitrary bytes. If these bytes (control characters) are displayed in logs or terminals, they can cause log forging or terminal injection.
**Prevention:** Always sanitize strings read from external binary formats before using them in logs, UIs, or internal identifiers, especially if they are expected to be ASCII/printable.
