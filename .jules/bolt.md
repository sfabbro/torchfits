## 2024-05-24 - FITS Header Parsing Optimization
**Learning:** FITS headers are padded with spaces to fill 2880-byte blocks, meaning after the `END` keyword, there can be thousands of bytes of empty space. Parsing these empty characters introduces significant overhead for large FITS files.
**Action:** When reading FITS header blocks, always break parsing loops early as soon as the `END     ` card is encountered to skip processing trailing padding.
