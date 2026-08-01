# CLI recipes

Shell workflows for common FITS jobs. Samples come from
`examples/_sample_data.py` (cached under `~/.cache/torchfits/samples/`).

```bash
pixi run python -c "from examples._sample_data import ensure_sample; print(ensure_sample('horsehead'))"
export HH=~/.cache/torchfits/samples/horsehead.fits
```

Or run the bundled script:

```bash
bash examples/cli/imstat_imarith.sh
```

Flags: every option is on `torchfits <cmd> --help` — see also [CLI guide](cli.md).

---

## Image inventory and stats

```bash
torchfits info "$HH"
torchfits stats "$HH" -e 0
torchfits header "$HH" -k OBJECT -k BITPIX
torchfits header "$HH" -k 'NAXIS*'
```

## Header edit

```bash
torchfits copy "$HH" /tmp/hh_edit.fits
torchfits setkey /tmp/hh_edit.fits -k OBJECT --value HORSEHEAD
torchfits setkey /tmp/hh_edit.fits -k "ESO DET CHIP1 ID" --value "42"
torchfits setkey /tmp/hh_edit.fits --rename OBJECT=TARGET
torchfits setkey /tmp/hh_edit.fits -k TEMPKEY --value 1
torchfits setkey /tmp/hh_edit.fits --delete TEMPKEY
torchfits header /tmp/hh_edit.fits -k TARGET
```

## Multi-file keyword table

```bash
torchfits header "$HH" /tmp/hh_edit.fits --keyword-table -k OBJECT -k NAXIS1
printf '%s\n' "$HH" /tmp/hh_edit.fits > /tmp/fits_list.txt
torchfits setkey @/tmp/fits_list.txt -k RUN --value demo --out-dir /tmp/keyed -J 0
```

## Image arithmetic (imarith-style)

```bash
# Scalar
torchfits arith "$HH" --op add --value 100 --out /tmp/hh_plus.fits
torchfits stats /tmp/hh_plus.fits -e 0

# Image–image (second path is operand B)
torchfits arith "$HH" /tmp/hh_plus.fits --op mul -o /tmp/hh_prod.fits

# Multi-file A × scalar (-J fans out across files)
# torchfits arith a.fits b.fits --op mul --value 2 --out-dir /tmp/scaled -J 0
```

## Cutout / copy

```bash
torchfits cutout "${HH}[101:256,101:256]" /tmp/hh_cut.fits
torchfits cutout "$HH" /tmp/hh_cut_box.fits -e 0 --box 100,100,256,256
torchfits copy "$HH" /tmp/hh_copy.fits
```

## Stretch via transform

Default constructors, or pass kwargs via `--name Class:key=val,...` (see
[CLI guide](cli.md)):

```bash
torchfits transform "$HH" --name LogStretch --out /tmp/hh_log.fits
torchfits transform "$HH" --name SqrtStretch --out /tmp/hh_sqrt.fits
torchfits transform "$HH" --name ZScaleNormalize --out /tmp/hh_z.fits
torchfits transform "$HH" --name PercentileClipNormalize:lower_pct=1.0,upper_pct=99.0 --out /tmp/hh_pct.fits
```

For multi-step parameterized pipelines, use the Python API
([Python workflows](python-workflows.md),
[Transform gallery](examples-transforms.md)).

## Compression

```bash
torchfits compress "$HH" /tmp/hh_packed.fits
torchfits decompress /tmp/hh_packed.fits /tmp/hh_unpacked.fits
```

## Verify checksums

```bash
torchfits verify "$HH"
```

## Tables and filter+export

```bash
pixi run python -c "from examples._sample_data import ensure_sample; print(ensure_sample('chandra_events'))"
export EV=~/.cache/torchfits/samples/chandra_events.fits
torchfits info "$EV"
torchfits table "$EV" -e 1 -n 5
torchfits convert "$EV" /tmp/events.parquet --to parquet -e 1
torchfits convert "$EV" /tmp/events.csv --to csv -e 1
# STILTS-like filter+export (predicate = table.read where=)
torchfits convert "$EV" -o /tmp/bright.parquet -e 1 -w "energy > 500" -c time,energy
```

## Color RGB preview (3 bands)

HorseHead is single-band — a grayscale `--bands 0,0,0` export is only a
smoke test. For a real Lupton RGB:

```bash
pixi run python examples/cli/make_rgb_demo.py /tmp
torchfits convert /tmp/_rgb_demo/r.fits /tmp/_rgb_demo/g.fits /tmp/_rgb_demo/b.fits \
  /tmp/rgb_demo.png --to png --q 6 --stretch 0.4
```

Tune `--q` / `--stretch` for contrast. Gallery asset:
`docs/assets/gallery/cli_rgb_demo.png`.

---

## Familiar-tool map

| Classic | torchfits |
|---------|-----------|
| `imstat` | `stats` |
| `imarith` (constant) | `arith` |
| `imcopy` | `copy` / `cutout` |
| `imheader` / `hedit` | `header` / `setkey` |
| `hselect` / `fitsort` | `header --keyword-table` |
| `imfunction` | `transform --name …` |
| `fitsinfo` | `info` |
| `fitsverify` (checksums) | `verify` (checksum keywords only; missing keywords = OK) |
| `fpack` / `funpack` | `compress` / `decompress` |
| `astconvertt` / STILTS `tpipe` | `convert` (+ `--where` / `--columns`) |

Full flags: [CLI guide](cli.md).
