# torchfits CLI

After installing `torchfits`, the `torchfits` command line tool
inspects and transforms FITS files directly from your shell, using the same high-performance C++ engine as the Python library.

**Help is always available:** run `torchfits --help` or `torchfits <cmd> --help` to see every available flag and option.

- **Inspection commands** (`info`, `header`, `verify`, `stats`, `table`, `probe`) read file structures, headers, and metadata from command-line paths, file lists, or standard input (`--stdin`).
- **File processing commands** (`copy`, `cutout`, `arith`, `convert`, `compress`, `decompress`, `setkey`, `transform`) execute transformations and write output files.

Common short flags:

| Short | Long | Notes |
|-------|------|-------|
| `-e` | `--hdu` | HDU index (e.g. `-e 0`, `-e 1,2`; avoids clashing with `-h` help) |
| `-f` | `--format` | Output format: `text` / `json` / `jsonl` |
| `-o` | `--out` | Output file path (also positional in copy/cutout/convert/compress) |
| `-w` / `-c` | `--where` / `--columns` | Table row filter condition and column selection |
| `-n` | `--rows` | Number of preview rows for `table` |
| `-k` | `--keyword` / `--key` | Header keyword filter in `header`, or target key in `setkey` |

`probe --header-bytes` controls the remote header preview byte range; `--timeout` sets the network timeout in seconds.

### Performance: Shell CLI vs Python API

Each shell command invocation initializes Python and loads the PyTorch runtime. For single inspection tasks or batch shell scripts, this startup time is negligible. However, for tight computational loops (e.g. reading thousands of cutouts in training loops), prefer using the in-process Python API ([Python workflows](python-workflows.md)).

## Install and help

```bash
pip install torchfits
torchfits --help
torchfits info --help
torchfits convert --help
```

## Quick examples

```bash
torchfits info science.fits
torchfits header science.fits                 # all HDUs, fitsheader-style text
torchfits header science.fits -k OBJECT -f json
torchfits verify science.fits
torchfits stats science.fits -e 0 -f jsonl
torchfits table catalog.fits -e 1 -n 5
torchfits cutout 'science.fits[100:256,100:256]' cutout.fits
torchfits cutout science.fits -o cutout.fits -e 0 --box 100,100,256,256
torchfits convert catalog.fits -o out.parquet -e 1
torchfits convert catalog.fits out.csv -e 1
torchfits convert catalog.fits -o filtered.parquet -e 1 -w "flux > 2" -c ra,dec,flux
torchfits convert r.fits g.fits b.fits -o rgb.png
torchfits copy science.fits -o science_copy.fits
torchfits compress science.fits -o science.fits.fz
torchfits compress *.fits --out-dir /tmp/fz -J 0   # -J = file workers
torchfits compress mef.fits --split hdu --out-dir /tmp/fz_hdus
torchfits arith a.fits b.fits --op mul -o product.fits
torchfits arith mef.fits --op mul --value 2 -e 0,1 -o mef_x2.fits
torchfits probe https://example.edu/file.fits --header-bytes 5760 --timeout 30
```

Multi-file / stdin inventory:

```bash
torchfits info a.fits b.fits c.fits -f jsonl
find . -name '*.fits' | torchfits info --stdin -f jsonl
printf '%s\n' *.fits | torchfits header --stdin --keyword-table -k OBJECT -k NAXIS1
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | success |
| 1 | files differ (`diff`) |
| 2 | usage error |
| 3 | I/O error |
| 4 | checksum verification failed (`verify`) |

## Subcommands

| Command | What it does |
|---------|----------------|
| `info` | list HDUs (type, shape, rows) |
| `header` | dump all HDUs (fitsheader-style text); `-k` filter; `--keyword-table` |
| `verify` | check `DATASUM` / `CHECKSUM` |
| `stats` | image min / max / mean |
| `table` | Arrow schema + preview rows |
| `cutout` | write a pixel box to a new FITS file |
| `convert` | table → Parquet/CSV/TSV/Arrow/FITS; filter with `--where`; Lupton RGB → PNG |
| `probe` | local = `info`; HTTP(S)/vos = header peek (`--header-bytes` / `--timeout`) |
| `diff` | compare two files (exit 1 if they differ) |
| `copy` | MEF-preserving FITS → FITS copy; multi-file `--out-dir` + `-J` |
| `arith` | image ±×÷ scalar **or** second image; multi-HDU / multi-file |
| `compress` / `decompress` | tile-compress or expand; `compress --algorithm`; `--out-dir`; `--split file|hdu` |
| `transform` / `cutout` | named transforms / pixel box; multi-file `--out-dir` + `-J` |
| `setkey` | set / rename / `--delete`; `@list` paths; `--out-dir` + `-J` |

### Multi-extension FITS (MEF)

Most commands walk **all HDUs** by default (including `header`). Narrow with
`-e 0,1,2` where that command accepts an HDU list. JSONL is one JSON object per
output item; for example, `header -f jsonl` emits header-card records, while
inventory commands emit file/HDU records.

### Parallelism / cores

Two knobs (orthogonal):

| Flag | Meaning | Default |
|------|---------|---------|
| `-j` / `--jobs` | PyTorch **intra-op** threads (`torch.set_num_threads`) | `0` = CPU count |
| `-J` / `--file-jobs` | **Python thread pool across files** | `0` = CPU count when ≥2 files, else 1 |

- **`-j`**: tensor work inside a file (e.g. `arith` stacked HDUs, `stats` reductions).
- **`-J`**: fan-out across input files for `compress` / `decompress` / `verify` /
  `stats` / `arith` / `copy` / `transform` / `cutout` / `setkey`. Each file
  worker caps ATen to 1 thread so CFITSIO I/O is not oversubscribed.
- **`compress` / `decompress`**:
  - `--algorithm` on compress (default `RICE_1`; also `GZIP_1`, `GZIP_2`,
    `HCOMPRESS_1`, …) — same strings as `write(..., compress=)`.
  - `--split file` (default): one output MEF per input (`-o` / `--out-dir`).
  - `--split hdu`: one output per **image** HDU
    (`{stem}_hdu00.fits`, `_hdu01`, … under `--out-dir`; required; width grows
    past two digits when needed). Narrow with `-e`. Non-image HDUs are skipped.
- **`arith`**: CFITSIO-style imarith — `--value` scalar **or** image B
  (`a.fits b.fits -o out` / `--operand2`); multi-HDU same-shape → stack+ATen;
  multi-file A via `--out-dir` + `-J`.
- **`stats`**: `min` / `max` / `mean` / `std` / `median` (population std).
- **`copy` / `transform` / `cutout`**: batch via `--out-dir` + `-J` (or
  `INPUT OUTPUT` / `-o` for one file).

### Output formats

Inventory commands (`info`, `header`, `verify`, `stats`, `table`, `probe`)
accept:

| Flag | Meaning |
|------|---------|
| (default) | human text |
| `-f json` / `--json` | JSON array |
| `-f jsonl` / `--jsonl` | one JSON object per line |

### `cutout`

Two syntaxes (pick one per invocation):

- **CFITSIO image section** on the path (1-based inclusive) — familiar from
  `imcopy` / CFITSIO:
  `torchfits cutout 'img.fits[10:100,20:200]' out.fits`
- **`--box x1,y1,x2,y2`** — 0-based half-open (same as `read_subset`):
  `torchfits cutout img.fits -o out.fits --box 9,19,100,200`

Supported: image pixel sections via path (`cutout` CLI / `read_tensor`).
Out of scope for this command: path HDU selectors (`file.fits[1]`),
binspec/histogram filenames, stacking a section with `--box`, and path
filters for catalogs (use `table.read(..., where=)`).

### `verify`

Checks **`DATASUM` / `CHECKSUM` only** (CFITSIO `ffvcks`). Covers checksum
keywords only; HEASARC `fitsverify` structural checks (mandatory keywords,
XTENSION rules, etc.) are out of scope.

Text output uses three labels:

| Label | Meaning | Exit code |
|-------|---------|-----------|
| `OK (no checksum keywords)` | HDU has no `DATASUM`/`CHECKSUM` — nothing to verify | 0 |
| `OK` | Checksums present and valid | 0 |
| `FAIL` | Checksums present but incorrect (corrupt) | 4 |

Files without checksum keywords exit **0** with label
`OK (no checksum keywords)` — there is nothing to verify (fitsverify-style
warning, not corruption). Add checksums from Python before verification:

```python
import torchfits

torchfits.write_checksums("science.fits", hdu=0)
```

```bash
torchfits verify science.fits
torchfits verify *.fits -f jsonl
```

JSON/JSONL output adds a `"status"` field (`"ok"`, `"no_checksums"`,
`"fail"`) alongside `"ok"`, `"datastatus"`, and `"hdustatus"`.

### `setkey`

Set, rename, or delete header keywords. Supports short cards, **HIERARCH** /
long names, `-e all` (or a comma list), `@list` path files, and multiple files
(`--out-dir` + `-J`). Edits use CFITSIO card update/delete (binary copy when
writing to a new path), so tile-compressed HDUs stay compressed.

```bash
torchfits setkey science.fits -k OBJECT --value NGC1234
torchfits setkey science.fits -k "ESO DET CHIP1 ID" --value "42" -e all
torchfits setkey *.fits --rename OBJECT=TARGET -e 0 --out-dir /tmp/edited
torchfits setkey @paths.txt --delete FOO --out-dir /tmp/edited -J 0
```

### `header`

Default text mode prints every HDU in fitsheader / listhead style:

```text
# HDU 0 (PRIMARY) in science.fits:
SIMPLE  =                    T / file does conform to FITS standard
BITPIX  =                  -32 / number of bits per data pixel
...

# HDU 1 (SCI) in science.fits:
XTENSION= 'IMAGE'              / IMAGE extension
```

Use `-e` to select HDUs, `-k` to filter keywords (shell-style wildcards like
`NAXIS*`), or `-f json` / `jsonl` for structured output. `--keyword-table`
prints a keyword table across many files (wildcards expand to matching columns):

```bash
torchfits header science.fits
torchfits header science.fits -e 1 -k OBJECT
torchfits header science.fits -k 'NAXIS*'
torchfits header *.fits --keyword-table -k OBJECT -k DATE-OBS
torchfits header *.fits --keyword-table -k BITPIX -f json
```

### `convert`

- **parquet** / **csv** / **tsv** / **arrow** / **fits** — export a table HDU
  (`-e`, default 1). Streaming writers keep large catalogs out-of-core when
  no filter is applied.
  - `--where` + optional `--columns` — filter+export (STILTS-like subset, not
    full STILTS). Same predicate syntax as `table.read(..., where=)`.
  - `csv` / `tsv` are for flat columns (nested / list columns need parquet or
    arrow).
  - `arrow` is Arrow IPC / Feather V2 (``.arrow``).
- **png** — Lupton asinh RGB preview from a file containing three image HDUs
  (`--bands 0,1,2`) or three band files. `--bands` selects HDU indices; it does
  not split planes from one 3-D cube. Writes PNG with torch + stdlib only (no
  Pillow dependency).

`--to` is optional when the output extension is unambiguous
(`.parquet`, `.csv`, `.tsv`/`.tab`, `.arrow`/`.feather`/`.ipc`, `.fits`, `.png`).

Defaults are for previews, not journal figures — retune stretch / Q per survey.

### `transform`

`--name` is a class from `torchfits.transforms.__all__`. Append
`:key=val,key2=val2` to pass constructor kwargs (values are parsed as
bool/int/float, else left as a string); unknown kwargs are rejected before
construction.

```bash
torchfits transform image.fits --name ArcsinhStretch -o out.fits
torchfits transform image.fits --name ArcsinhStretch:a=2.0 -o out.fits
torchfits transform image.fits --name PercentileClipNormalize:lower_pct=1.0,upper_pct=99.0 -o out.fits
```

### `probe`

- **Local paths** — same inventory as `info` (`-e` selects HDUs).
- **HTTP(S)** — range-fetch primary header (`--header-bytes`, `--timeout`); `-e` is
  ignored for remote peeks (primary only). Follows redirects with SSRF checks;
  optional `TORCHFITS_HTTP_AUTHORIZATION` / `TORCHFITS_HTTP_TOKEN`.
- **`vos:` / `vault:` / `vos://`** — optional; install the `vos` package.
  Short `vos:<user>/...` and `vault:<user>/...` map to
  `vos://cadc.nrc.ca~vault/<user>/...`. Auth uses the client’s normal config.

```bash
torchfits probe science.fits
torchfits probe https://example.edu/data.fits --header-bytes 5760 --timeout 15 -f json
torchfits probe vos:alice/data/sample.fits
```

Archive *search* (CAOM / `astquery`-style queries) is out of scope.

### `cutout`

Pixel box extraction. HTTP(S) **uncompressed 2D** inputs use Range GETs;
compressed remotes download into the cache first (same as `read_subset`).

## Familiar-tool mapping

| torchfits command | Closest classic tools | Purpose & Description |
|---|---|---|
| `info` | [`fitsinfo`](https://docs.astropy.org/en/stable/io/fits/usage/scripts.html#fitsinfo) | Overview of HDU extensions, dimensions, and data types |
| `header` | [`fitsheader`](https://docs.astropy.org/en/stable/io/fits/usage/scripts.html#fitsheader), [`dfits` / `fitsort`](https://www.eso.org/sci/software/eclipse/eug/eug/node13.html) | Dump header cards; filter by keyword; multi-file summary tables |
| `verify` | [`fitscheck`](https://docs.astropy.org/en/stable/io/fits/usage/scripts.html#fitscheck), [`fitsverify`](https://heasarc.gsfc.nasa.gov/docs/software/ftools/fitsverify/) | Check `DATASUM` and `CHECKSUM` keyword integrity |
| `stats` | [`imstat`](https://iraf.net/irafdocs/imstat.php), [`aststatistics`](https://www.gnu.org/software/gnuastro/manual/html_node/Invoking-aststatistics.html) | Compute min, max, mean, standard deviation, and median pixel values |
| `table` | [`asttable`](https://www.gnu.org/software/gnuastro/manual/html_node/Invoking-asttable.html), `tablist` | Preview binary/ASCII table schema and row values |
| `cutout` | [`astcrop`](https://www.gnu.org/software/gnuastro/manual/html_node/Invoking-astcrop.html), CFITSIO sections | Extract sub-regions using pixel ranges (`[x1:x2, y1:y2]`) or bounding boxes |
| `convert` | [`astconvertt`](https://www.gnu.org/software/gnuastro/manual/html_node/Invoking-astconvertt.html), [`STILTS`](https://www.star.bristol.ac.uk/~mbt/stilts/) | Export tables to Parquet/CSV/Arrow with SQL filters; render 3-band Lupton RGB PNGs |
| `copy` | [`fitscopy` / `imcopy`](https://heasarc.gsfc.nasa.gov/fitsio/fpack/) | Lossless copy of single or multi-extension FITS files |
| `arith` | [`imarith`](https://iraf.net/irafdocs/imarith.php) | Perform scalar or image-to-image addition, subtraction, multiplication, division |
| `compress` / `decompress` | [`fpack` / `funpack`](https://heasarc.gsfc.nasa.gov/fitsio/fpack/) | Lossless / lossy tile compression (Rice, Gzip, Hcompress) and expansion |
| `transform` | `imfunction` | Apply astronomical stretches (Arcsinh, Sqrt, Log) and ZScale / percentile scaling |
| `setkey` | `hedit`, `modhead` | Insert, update, rename, or delete header keywords across single or multiple files |
| `probe` | HTTP Range peek | Inspect remote FITS headers over HTTP(S) or VOSpace without downloading whole files |

For practical examples, see [CLI recipes](cli-recipes.md).

## Scripting notes

- No prompts; stable exit codes.
- Prefer `-f json` / `jsonl` (or `--json` / `--jsonl`) for automation.
- GPU tensors are staged through host memory before any FITS write (same as the
  Python API — not GPUDirect).
