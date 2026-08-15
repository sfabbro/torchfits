# torchfits CLI

After installing `torchfits`, the `torchfits` command line interface lets you
inspect, manipulate, and transform FITS files directly from your shell, backed by the same high-performance C++ engine and vendored CFITSIO as the Python library.

Run `torchfits --help` or `torchfits <subcommand> --help` anytime for the complete list of options.

---

## Quick Tour

```bash
# Inspection & metadata
torchfits info science.fits
torchfits header science.fits -k OBJECT -k 'NAXIS*'
torchfits header *.fits --keyword-table -k OBJECT -k FILTER
torchfits stats science.fits -e 0 -f json
torchfits table catalog.fits -e 1 -n 5
torchfits verify science.fits

# File comparison & copies
torchfits diff file_a.fits file_b.fits
torchfits copy science.fits science_copy.fits

# Transformations & arithmetic
torchfits cutout 'science.fits[100:256,100:256]' cutout.fits
torchfits arith science.fits --op mul --value 2.0 -o doubled.fits
torchfits arith science.fits flat.fits --op div -o calibrated.fits
torchfits transform science.fits --name ArcsinhStretch -o stretched.fits

# Format conversions & previews
torchfits convert catalog.fits catalog.parquet -e 1
torchfits convert catalog.fits bright.parquet -e 1 -w "MAG_G < 18.0" -c RA,DEC,MAG_G
torchfits convert r.fits g.fits b.fits -o rgb.png --to png

# Compression
torchfits compress science.fits science.fits.fz
torchfits decompress science.fits.fz science_decomp.fits

# Header modification
torchfits setkey science.fits -k OBJECT --value "NGC 1234"
torchfits setkey science.fits --rename OLDNAME=NEWNAME

# Remote peeking (HTTP/HTTPS/VOSpace)
torchfits probe https://example.edu/survey/image.fits --header-bytes 5760
```

---

## Global Options & Flags

Common flags shared across subcommands:

| Flag | Name | Purpose | Default |
|---|---|---|---|
| `-e` | `--hdu` | Target HDU index (e.g. `-e 0`, `-e 1,2`, or `-e all` for `setkey`) | All HDUs or 0/1 depending on command |
| `-f` | `--format` | Output format: `text`, `json`, or `jsonl` | `text` |
| `-o` | `--out` | Output file path | Positional argument or stdout |
| `--out-dir` | `--out-dir` | Output directory for batch file processing | Current directory |
| `--stdin` | `--stdin` | Read input file paths from standard input | `False` |
| `-j` | `--jobs` | PyTorch intra-op CPU threads (`torch.set_num_threads`) | `0` (all CPU cores) |
| `-J` | `--file-jobs` | Parallel worker thread pool across multiple files | `0` (all CPU cores if $\ge 2$ files, else 1) |

### Parallelism & Multi-Core Execution

The CLI provides two orthogonal parallelism knobs:

1. **`-j` / `--jobs` (Intra-file parallelism):** Controls the number of PyTorch internal threads used for array computations (e.g. image arithmetic reductions, statistics calculations).
2. **`-J` / `--file-jobs` (Inter-file batch parallelism):** Spawns a thread pool to process multiple independent FITS files concurrently. When batching files with `-J`, each worker automatically sets intra-op threads to 1 to prevent CPU oversubscription.

### Performance: Shell CLI vs Python API

Each command invocation starts the Python interpreter and loads the PyTorch runtime. For shell automation, scripting, and file management, this startup overhead is negligible. For tight computational loops (such as dataset iterators and deep learning training pipelines), use the in-process Python API ([Python workflows](python-workflows.md)).

---

## Exit Codes

| Code | Meaning | Example |
|---|---|---|
| `0` | Success | Command completed successfully |
| `1` | Difference found | `torchfits diff` found header or image discrepancies |
| `2` | Usage error | Missing arguments, invalid flags, or unknown syntax |
| `3` | I/O error | File not found, permission denied, or invalid FITS structure |
| `4` | Checksum verification failure | `torchfits verify` detected invalid `DATASUM` or `CHECKSUM` |

---

## Subcommand Reference

| Subcommand | Description |
|---|---|
| [`info`](#info) | Summary of HDUs (extension index, name, type, dimensions, data type) |
| [`header`](#header) | Display header cards, filter keywords, or build multi-file summary tables |
| [`verify`](#verify) | Validate `DATASUM` and `CHECKSUM` integrity across HDUs |
| [`diff`](#diff) | Compare header keywords and pixel statistics between two FITS files |
| [`stats`](#stats) | Compute image statistics (min, max, mean, standard deviation, median) |
| [`table`](#table) | Inspect binary/ASCII table schemas and preview rows |
| [`cutout`](#cutout) | Extract sub-regions using pixel coordinates or CFITSIO sections |
| [`convert`](#convert) | Export tables to Parquet, CSV, TSV, Arrow, or render 3-band RGB PNGs |
| [`copy`](#copy) | Lossless copy of single or multi-extension FITS files |
| [`arith`](#arith) | Perform scalar or image-to-image arithmetic (+, -, *, /) |
| [`compress`](#compress) | Tile-compress images using Rice, Gzip, or Hcompress algorithms |
| [`decompress`](#decompress) | Uncompress tile-compressed FITS files to standard FITS |
| [`transform`](#transform) | Apply astronomical stretch and normalization transforms |
| [`setkey`](#setkey) | Insert, update, rename, or delete header keywords |
| [`probe`](#probe) | Peek at local or remote HTTP(S) / VOSpace FITS headers without downloading |

---

### `info`

Lists all HDU extensions in one or more FITS files, showing HDU index, name, extension type, dimensions, and data type.

```bash
# Inspect a single file
torchfits info science.fits

# Inspect multiple files with JSON Lines output
torchfits info file1.fits file2.fits -f jsonl

# Read paths from standard input
find . -name "*.fits" | torchfits info --stdin -f jsonl
```

---

### `header`

Dumps header cards from one or more HDUs in standard FITS card format, with support for keyword filtering and multi-file summary catalogs.

```bash
# Print all headers in all HDUs
torchfits header science.fits

# Select specific HDUs
torchfits header science.fits -e 0,1

# Filter for specific keywords (supports wildcards)
torchfits header science.fits -k OBJECT -k 'NAXIS*'

# Export header cards to structured JSON
torchfits header science.fits -e 0 -f json

# Build a tabular summary across multiple files
torchfits header *.fits --keyword-table -k OBJECT -k FILTER -k EXPTIME
```

---

### `verify`

Validates FITS `DATASUM` and `CHECKSUM` keywords using CFITSIO's fast verification engine.

```bash
# Verify checksums for all HDUs
torchfits verify science.fits

# Batch verify multiple files in parallel
torchfits verify *.fits -J 0 -f jsonl
```

Output statuses:

- `OK`: Checksums are present and valid.
- `OK (no checksum keywords)`: File has no checksum keywords (returns exit code 0).
- `FAIL`: Checksum does not match data (returns exit code 4).

---

### `diff`

Compares two FITS files HDU by HDU. Compares all header keywords (ignoring transient `CHECKSUM`/`DATASUM`) and evaluates pixel shapes, minimums, maximums, and means for image extensions. Returns exit code `0` if identical, `1` if different.

```bash
# Compare two files
torchfits diff image_v1.fits image_v2.fits
```

---

### `stats`

Computes pixel statistics for image HDUs: minimum, maximum, mean, standard deviation, and median.

```bash
# Compute statistics for primary HDU
torchfits stats science.fits -e 0

# Output in JSON format
torchfits stats science.fits -e 0 -f json

# Process multiple files in parallel
torchfits stats *.fits -e 0 -J 0 -f jsonl
```

---

### `table`

Inspects binary or ASCII table extensions, displaying the PyArrow schema, column types, and a preview of rows.

```bash
# Preview table schema and first 5 rows
torchfits table catalog.fits -e 1 -n 5

# Select specific columns
torchfits table catalog.fits -e 1 -c RA,DEC,FLUX -n 10

# Output rows in JSON format
torchfits table catalog.fits -e 1 -n 5 -f json
```

---

### `cutout`

Extracts a sub-region from an image HDU and writes it directly to a new FITS file, preserving header metadata and WCS reference pixels.

Supports two coordinate formats:

1. **CFITSIO Section Syntax (1-based inclusive):**
   ```bash
   torchfits cutout 'science.fits[101:256,101:256]' cutout.fits
   ```
2. **Bounding Box Syntax (0-based half-open `x1,y1,x2,y2`):**
   ```bash
   torchfits cutout science.fits -o cutout.fits -e 0 --box 100,100,256,256
   ```

```bash
# Batch extract cutouts across multiple files
torchfits cutout *.fits --box 100,100,256,256 --out-dir /tmp/cutouts -J 0
```

---

### `convert`

Converts FITS binary/ASCII tables to modern data science formats (Parquet, CSV, TSV, Arrow IPC) with optional in-engine row filtering, or renders 3-band RGB PNG images.

#### Table Conversion & Filtering
```bash
# Convert to Apache Parquet (format inferred from .parquet suffix)
torchfits convert catalog.fits catalog.parquet -e 1

# Convert to CSV selecting specific columns
torchfits convert catalog.fits -o catalog.csv -e 1 -c RA,DEC,MAG_G

# Pushdown row filtering (STILTS-style SQL predicate)
torchfits convert catalog.fits -o filtered.parquet -e 1 -w "MAG_G < 19.5 AND DEC > 0" -c RA,DEC,MAG_G
```

#### 3-Band Color RGB PNG Rendering
```bash
# Combine 3 separate band files into a Lupton RGB PNG
torchfits convert r.fits g.fits b.fits -o preview.png --to png --q 8.0 --stretch 0.5

# Render from a single MEF containing 3 image HDUs
torchfits convert multi_band.fits -o preview.png --bands 0,1,2 --to png
```

---

### `copy`

Performs an exact, lossless binary copy of a FITS file. Supports batch operations across directories with thread pool concurrency.

```bash
# Copy single file
torchfits copy science.fits backup.fits

# Batch copy multiple files
torchfits copy *.fits --out-dir /backup/fits -J 0
```

---

### `arith`

Performs scalar and image-to-image arithmetic (`add`, `sub`, `mul`, `div`). Same-shape HDUs are stacked for accelerated PyTorch vector execution.

```bash
# Add scalar value
torchfits arith science.fits --op add --value 50.0 -o offset.fits

# Multiply image by calibration factor
torchfits arith science.fits --op mul --value 1.25 -o scaled.fits

# Image-to-image subtraction (science - dark)
torchfits arith science.fits dark.fits --op sub -o dark_subtracted.fits

# Batch process multiple files against a shared scalar
torchfits arith *.fits --op mul --value 2.0 --out-dir /tmp/scaled -J 0
```

---

### `compress`

Tile-compresses image HDUs using CFITSIO's compression architecture. Preserves non-image HDUs and table extensions.

Supported algorithms: `RICE_1` (default), `GZIP_1`, `GZIP_2`, `HCOMPRESS_1`, `PLIO_1`.

```bash
# Compress with default RICE_1
torchfits compress science.fits science.fits.fz

# Compress using GZIP algorithm
torchfits compress science.fits science.fits.fz --algorithm GZIP_1

# Split MEF file into individual compressed HDU files
torchfits compress mef.fits --split hdu --out-dir /tmp/compressed_hdus

# Batch compress all files in a directory in parallel
torchfits compress *.fits --out-dir /tmp/compressed -J 0
```

---

### `decompress`

Expands tile-compressed `.fits.fz` files back into standard uncompressed FITS files.

```bash
# Decompress single file
torchfits decompress science.fits.fz science.fits

# Batch decompress in parallel
torchfits decompress *.fits.fz --out-dir /tmp/uncompressed -J 0
```

---

### `transform`

Applies any astronomical normalization or stretch transform from `torchfits.transforms` directly from the shell.

Supported transforms include: `ArcsinhStretch`, `LogStretch`, `SqrtStretch`, `ZScaleNormalize`, `RobustNormalize`, `BackgroundSubtract`, `PercentileClipNormalize`, `MinMaxNormalize`, `GlobalScalarNorm`.

```bash
# Apply Arcsinh stretch
torchfits transform science.fits --name ArcsinhStretch -o stretched.fits

# Apply ZScale normalization
torchfits transform science.fits --name ZScaleNormalize -o zscale.fits

# Pass custom parameters via Name:key=val syntax
torchfits transform science.fits --name PercentileClipNormalize:lower_pct=2.0,upper_pct=98.0 -o clipped.fits
torchfits transform science.fits --name ArcsinhStretch:a=0.05 -o custom_arcsinh.fits

# Batch transform multiple files
torchfits transform *.fits --name LogStretch --out-dir /tmp/log_images -J 0
```

---

### `setkey`

Inserts, updates, renames, or deletes header keywords in place via CFITSIO card updates. Preserves tile compression on compressed files.

```bash
# Set or update keyword
torchfits setkey science.fits -k OBJECT --value "NGC 1234"

# Set keyword with comment
torchfits setkey science.fits -k FILTER --value "g" --comment "SDSS g-band filter"

# Set HIERARCH / long keyword
torchfits setkey science.fits -k "ESO DET CHIP1 ID" --value "42"

# Apply across all HDUs
torchfits setkey science.fits -k OBSERVER --value "astronomer" -e all

# Rename a keyword card
torchfits setkey science.fits --rename OBJECT=TARGET

# Delete a keyword card
torchfits setkey science.fits --delete TEMPKEY

# Batch update files from a text file list (@list syntax)
torchfits setkey @file_list.txt -k PROCESSED --value "TRUE" --out-dir /tmp/updated -J 0
```

---

### `probe`

Inspects local files or peeks at remote FITS headers over HTTP(S) and VOSpace using HTTP Range requests without downloading the full file.

```bash
# Probe local file
torchfits probe science.fits

# Peek at remote HTTP header (fetches only 5760 header bytes)
torchfits probe https://example.edu/survey/galaxy.fits --header-bytes 5760 --timeout 15 -f json

# Probe CADC VOSpace URI
torchfits probe vos:username/data/sample.fits
```

---

## Familiar-Tool Mapping

| torchfits command | Closest classic tools | Purpose & Description |
|---|---|---|
| `info` | [`fitsinfo`](https://docs.astropy.org/en/stable/io/fits/usage/scripts.html#fitsinfo) (Astropy) | Overview of HDU extensions, dimensions, and data types |
| `header` | [`fitsheader`](https://docs.astropy.org/en/stable/io/fits/usage/scripts.html#fitsheader) (Astropy), [`dfits` / `fitsort`](https://www.eso.org/sci/software/eclipse/eug/eug/node13.html) (ESO Eclipse) | Dump header cards; filter by keyword; multi-file summary tables |
| `verify` | [`fitscheck`](https://docs.astropy.org/en/stable/io/fits/usage/scripts.html#fitscheck) (Astropy), [`fitsverify`](https://heasarc.gsfc.nasa.gov/docs/software/ftools/fitsverify/) (NASA HEASARC) | Check `DATASUM` and `CHECKSUM` keyword integrity |
| `diff` | [`fitsdiff`](https://docs.astropy.org/en/stable/io/fits/usage/scripts.html#fitsdiff) (Astropy) | Compare headers and image statistics between two FITS files |
| `stats` | [`imstat`](https://iraf.net/irafdocs/imstat.php) (IRAF), [`aststatistics`](https://www.gnu.org/software/gnuastro/manual/html_node/Invoking-aststatistics.html) (Gnuastro) | Compute min, max, mean, standard deviation, and median pixel values |
| `table` | [`asttable`](https://www.gnu.org/software/gnuastro/manual/html_node/Invoking-asttable.html) (Gnuastro), `tablist` (NASA HEASARC FTOOLS) | Preview binary/ASCII table schema and row values |
| `cutout` | [`astcrop`](https://www.gnu.org/software/gnuastro/manual/html_node/Invoking-astcrop.html) (Gnuastro), CFITSIO image sections | Extract sub-regions using pixel ranges (`[x1:x2, y1:y2]`) or bounding boxes |
| `convert` | [`astconvertt`](https://www.gnu.org/software/gnuastro/manual/html_node/Invoking-astconvertt.html) (Gnuastro), [`STILTS`](https://www.star.bristol.ac.uk/~mbt/stilts/) (Starlink) | Export tables to Parquet/CSV/Arrow with SQL filters; render 3-band Lupton RGB PNGs |
| `copy` | [`fitscopy` / `imcopy`](https://heasarc.gsfc.nasa.gov/fitsio/fpack/) (CFITSIO) | Lossless copy of single or multi-extension FITS files |
| `arith` | [`imarith`](https://iraf.net/irafdocs/imarith.php) (IRAF) | Perform scalar or image-to-image addition, subtraction, multiplication, division |
| `compress` / `decompress` | [`fpack` / `funpack`](https://heasarc.gsfc.nasa.gov/fitsio/fpack/) (NASA HEASARC) | Lossless / lossy tile compression (Rice, Gzip, Hcompress) and expansion |
| `transform` | [`imfunction`](https://iraf.net/irafdocs/imfunction.php) (IRAF) | Apply astronomical stretches (Arcsinh, Sqrt, Log) and ZScale / percentile scaling |
| `setkey` | [`hedit`](https://iraf.net/irafdocs/hedit.php) (IRAF), `modhead` (WCSTools) | Insert, update, rename, or delete header keywords across single or multiple files |
| `probe` | HTTP Range peek | Inspect remote FITS headers over HTTP(S) or VOSpace without downloading whole files |

For practical examples, see [CLI recipes](cli-recipes.md).

## Scripting notes

- No prompts; stable exit codes.
- Prefer `-f json` / `jsonl` (or `--json` / `--jsonl`) for automation.
- GPU tensors are staged through host memory before any FITS write (same as the
  Python API — not GPUDirect).
