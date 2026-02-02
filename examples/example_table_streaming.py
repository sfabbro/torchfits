"""
Example: Stream FITS tables in chunks.
"""

import torchfits


def main():
    # Replace with your FITS table path
    path = "catalog.fits"

    for chunk in torchfits.stream_table(path, hdu=1, chunk_rows=10000):
        # chunk is a dict of column tensors (or lists for VLA)
        print(f"Columns: {list(chunk.keys())[:5]}")
        break


if __name__ == "__main__":
    main()
