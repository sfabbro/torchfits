Main objective: reading, writing FITS format files directly with PyTorch, faster and more user-friendly than any other methods.

## Files to be considered:
- Tables from 1000 rows to 200M or more rows, any data type, any number of columns
- Images from 10x10 cutouts or files, to 20,000x20,000 pixels and more
- Spectra, typically 1D arrays, but they could be with inverse variance and mask arrays
- Data cubes from many IFU instruments, radio astronomy, can be very big as well
- MEF which could be combinations of images and tables, such as HSC
- header only files
- list of many files of the types above.

## Features

* Fast FITS I/O: uses a highly optimized C++ backend (powered by `cfitsio`) for rapid data access,  making the smartest use of low-level cfitsio routines
* Direct PyTorch Tensors: Reads FITS data directly into PyTorch tensors, avoiding unnecessary data copies.
* Flexible Cutout Reading: Supports CFITSIO-style cutout strings (e.g., `'myimage.fits[1][10:20,30:40]'`).
*Automatic Data Type Handling:  Automatically determines the correct PyTorch data type.
* WCS Support: Includes functions for world-to-pixel and pixel-to-world coordinate transformations using wcslib, same thing for spectra wavelength, datacubes, etc...
* Header Access: Provides functions to access the full FITS header as a Python dictionary, or to retrieve individual header keyword values.
* HDU Information: Functions to get the number of HDUs, HDU types, and image dimensions, select HDU by name or number
* Designed for PyTorch Datasets: The API is designed to integrate seamlessly with PyTorch's `Dataset` and `DataLoader` classes, including distributed data loading, from many FITS files from many types. Randomization between cutouts from all these files, which can be MEFs.
* work for remote or cloud-hosted files directly
* Have intelligent caching
* Read from and write to image, binary, and ascii table extensions.
* Read arbitrary subsets of table columns and rows without loading all the data to memory (any device)
* Read image subsets without reading the whole image. Write subsets to existing images.
* Write and read variable length table columns.
* Read images and tables using notation similar to pytorch tensors. This is like a more powerful memmap, since it is column-aware for tables.
* Append rows to an existing table. Delete row sets and row ranges. Resize tables, or insert rows.
* Query the columns and rows in a table.
* Read and write header keywords.
* Read and write images in tile-compressed format (RICE,GZIP,PLIO,HCOMPRESS) (using cfitsio)
* Read/write gzip files directly. Read unix compress (.Z,.zip) and bzip2 (.bz2) files.
* TDIM information is used to return array columns in the correct shape.
* Write and read string table columns, including array columns of arbitrary shape.
* Read and write complex, bool (logical), unsigned integer, signed bytes types.
* Write checksums into the header and verify them.
* Insert new columns into tables in-place.
* Iterate over rows in a table. Data are buffered for efficiency.
* Follow astropy.io.fits user-facing [API](https://docs.astropy.org/en/latest/io/fits/api/index.html) but for pytorch, and more performant across the board.
* Takes advantage of PyTorch data types, in particular for 16-bit which can be common in raw sensor images
* when using 16-bits for FLOATS, include option for proper scaling (BITPIX/BSCALE/BZERO) even if compressed, a bit like 
* IS ALWAYS AT LEAST AS FAST as astropy to numpy, or fitsio to numpy
* IS ALWAYS MUCH FASTER than conversion from/to astropy or fitsio to numpy to pytorch