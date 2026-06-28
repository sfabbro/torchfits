# scripts/scratch/

One-off diagnostic / scaffolding scripts that were used to investigate
or verify behaviour during development. They are **not production
code**, are not invoked from any of the documented entry points
(`benchmarks/`, `tests/`, `pixi.toml` tasks), and are kept in the
repo as audit material so future parity-debugging can reuse the byte /
header dumps.

Each script in this directory served a specific purpose during the
COMPLEX / BIT / STRING `update_rows` parity work:

| script | purpose |
|---|---|
| `diag_header.py` | Dump the FITS BinTableHDU header (TTYPE, TFORM) for a freshly-written file so an on-disk write can be compared against what the schema declared. |
| `diag_string_bytes_v2.py` | Working `diag_string_bytes` (v1 removed) that dumps raw on-disk bytes for the `NAME` row section to verify the writer output. |
| `apply_routing_fix.py` | Interim scratch used to chase a phantom `stride(1) == 4` bug in the mmap path (turned out to be a fitsio-upstream-reader misdecode). |
| `fix_string_stride.py` | Interim scratch, sibling to `apply_routing_fix.py`. |

If you arrive here to debug a FITS write that looks wrong, run a
diagnostic script with `--help` (most have usage examples in their
`__main__` block) before reaching for stdlib `struct` unpacking.
