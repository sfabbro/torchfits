# Data Access Guide (Draft)

Covers reading/writing images, tables, cubes, MEFs, and compressed FITS.

Sections to implement:

1. Opening files & listing HDUs
2. Selecting HDUs (by index, EXTNAME)
3. Reading images → torch.Tensor (memory mapping, cutouts)
4. Reading tables → FitsTable → tensor columns
5. Variable-length arrays
6. Compressed images (Rice, GZIP, HCOMPRESS)
7. Random groups (legacy)
8. WCS utilities
