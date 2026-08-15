# CLI Recipes

Practical, copy-pasteable shell recipes for common FITS workflows using the `torchfits` command-line interface.

!!! tip "Getting Sample Data"
    To download the sample files used in these recipes (such as `horsehead.fits` and `chandra_events.fits`), run:
    ```bash
    bash scripts/fetch_example_samples.sh
    ```
    Or use any of your own FITS files in place of the filenames below.

---

## 1. Inspecting Images and Headers

```bash
# Print summary of HDUs (extensions, dimensions, data types)
torchfits info science.fits

# Compute pixel statistics (min, max, mean, standard deviation)
torchfits stats science.fits -e 0

# Print all header keywords
torchfits header science.fits

# Filter header for specific keywords (supports wildcards)
torchfits header science.fits -k OBJECT -k BITPIX -k 'NAXIS*'

# Export header cards to JSON or JSON Lines
torchfits header science.fits -k OBJECT -f json
```

---

## 2. Editing Headers

```bash
# Set or update a keyword value
torchfits setkey science.fits -k OBJECT --value "M31"

# Add a long HIERARCH keyword
torchfits setkey science.fits -k "ESO DET CHIP1 ID" --value "42"

# Rename a keyword card
torchfits setkey science.fits --rename OBJECT=TARGET

# Delete a keyword
torchfits setkey science.fits --delete TEMPKEY

# Batch edit multiple files across CPU cores (-J 0 uses all available cores)
torchfits setkey *.fits -k OBSERVER --value "astronomer" --out-dir /tmp/edited -J 0
```

---

## 3. Multi-File Header Catalog & Tables

Extract keywords across many files into a single tabular view:

```bash
# Generate a keyword summary table across all matching FITS files
torchfits header *.fits --keyword-table -k OBJECT -k FILTER -k EXPTIME

# Export the keyword catalog directly to JSON
torchfits header *.fits --keyword-table -k OBJECT -k NAXIS1 -f json
```

---

## 4. Image Arithmetic (`imarith`-style)

Perform scalar and image-to-image math directly from the shell:

```bash
# Add a scalar background offset
torchfits arith science.fits --op add --value 100 -o /tmp/science_offset.fits

# Multiply an image by a calibration factor
torchfits arith science.fits --op mul --value 1.5 -o /tmp/science_scaled.fits

# Image-to-image multiplication (science.fits × flat.fits)
torchfits arith science.fits flat.fits --op mul -o /tmp/science_calibrated.fits

# Image-to-image subtraction (science.fits - dark.fits)
torchfits arith science.fits dark.fits --op sub -o /tmp/science_sub.fits

# Batch process multiple files with parallel workers (-J 0)
torchfits arith raw_*.fits --op mul --value 2.0 --out-dir /tmp/scaled -J 0
```

---

## 5. Cutouts and Subsets

Extract specific pixel regions from 2D images or data cubes:

```bash
# Using 1-based inclusive pixel range (CFITSIO / DS9 syntax)
torchfits cutout 'science.fits[101:256,101:256]' /tmp/cutout.fits

# Using 0-based half-open bounding box (x1, y1, x2, y2)
torchfits cutout science.fits -o /tmp/cutout.fits -e 0 --box 100,100,256,256

# Make a lossless copy of an entire FITS file
torchfits copy science.fits /tmp/science_copy.fits
```

---

## 6. Astronomical Stretches & Normalizations

Apply non-linear contrast scaling and normalizations:

```bash
# Logarithmic stretch
torchfits transform science.fits --name LogStretch -o /tmp/science_log.fits

# Square-root stretch
torchfits transform science.fits --name SqrtStretch -o /tmp/science_sqrt.fits

# IRAF-style ZScale normalization
torchfits transform science.fits --name ZScaleNormalize -o /tmp/science_zscale.fits

# Percentile clipping (e.g. 1st to 99th percentile)
torchfits transform science.fits --name PercentileClipNormalize:lower_pct=1.0,upper_pct=99.0 -o /tmp/science_clipped.fits
```

---

## 7. Tile Compression & Decompression (`fpack` / `funpack`)

```bash
# Compress using RICE_1 tile compression (creates .fits.fz)
torchfits compress science.fits /tmp/science.fits.fz

# Compress using GZIP or HCOMPRESS algorithms
torchfits compress science.fits /tmp/science.fits.fz --algorithm GZIP_1

# Decompress a tile-compressed file back to standard FITS
torchfits decompress /tmp/science.fits.fz /tmp/science_decompressed.fits

# Batch compress all FITS files in a directory using all CPU cores
torchfits compress *.fits --out-dir /tmp/compressed_dir -J 0
```

---

## 8. Table Preview, Filtering, and Conversion

Convert FITS binary/ASCII tables to modern data science formats with optional pushdown filtering:

```bash
# Preview table columns and the first 5 rows
torchfits table catalog.fits -e 1 -n 5

# Convert table HDU to Apache Parquet
torchfits convert catalog.fits /tmp/catalog.parquet -e 1

# Convert table HDU to CSV (selecting specific columns)
torchfits convert catalog.fits -o /tmp/catalog.csv -e 1 -c RA,DEC,MAG_G

# Filter and export in a single pass (STILTS-style SQL predicate)
torchfits convert catalog.fits -o /tmp/bright_stars.parquet -e 1 -w "MAG_G < 18.0 AND DEC > 0" -c RA,DEC,MAG_G
```

---

## 9. 3-Band Color RGB Rendering

Combine three single-band FITS images into a publication-quality Lupton RGB PNG:

```bash
# Combine red, green, and blue band images into an RGB PNG preview
torchfits convert r.fits g.fits b.fits -o rgb_preview.png --to png --q 6 --stretch 0.4
```

---

## 10. Checksum Verification

Verify FITS `DATASUM` and `CHECKSUM` integrity:

```bash
# Verify integrity of all HDUs in a file
torchfits verify science.fits

# Batch verify all files in a folder and output status in JSON Lines format
torchfits verify *.fits -f jsonl
```

---

For the classic-tool mapping and complete flag reference, see the [CLI guide](cli.md).
