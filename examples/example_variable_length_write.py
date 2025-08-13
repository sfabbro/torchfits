"""
Example: Writing a variable-length array table and reading it back.

This creates a single-column binary table where each row is a 1D array of
variable length. Useful for spectra per-object, ragged time series, etc.
"""

import torch

import torchfits as tf


def main():
    # Create ragged 1D tensors (different lengths)
    arrays = [
        torch.linspace(0, 1, 10, dtype=torch.float32),
        torch.linspace(0, 1, 25, dtype=torch.float32),
        torch.linspace(0, 1, 5, dtype=torch.float32),
    ]

    # Write as a variable-length array table
    tf.write_variable_length_array(
        "varlen_example.fits",
        arrays,
        header={"EXTNAME": "VLA"},
        overwrite=True,
    )

    # Read back header and confirm
    hdr = tf.get_header("varlen_example.fits", hdu=1)
    print("Wrote variable-length array table:")
    print(f"  EXTNAME: {hdr.get('EXTNAME')}")
    print(f"  TFIELDS: {hdr.get('TFIELDS')}")
    print(f"  TFORM1:  {hdr.get('TFORM1')}")

    # Reading variable-length arrays may not be supported on all builds.
    # Try torchfits.read first; if it fails, fall back to astropy if present.
    try:
        table, _ = tf.read("varlen_example.fits", hdu=1)
        col = hdr.get("TTYPE1", "ARRAY_DATA")
        if isinstance(table, dict) and col in table:
            print(f"Read column '{col}' with {len(table[col])} rows")
        else:
            print(
                "Variable-length table read returned unexpected format; inspect manually."
            )
    except Exception as e:
        print(f"torchfits.read failed on variable-length array table: {e}")
        try:
            from astropy.io import fits  # optional verification

            with fits.open("varlen_example.fits") as hdul:
                data = hdul[1].data
                print(f"astropy verified rows: {len(data)}")
        except Exception:
            print("Skipping data verification; install astropy to inspect rows.")


if __name__ == "__main__":
    main()
