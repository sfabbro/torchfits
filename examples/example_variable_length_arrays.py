from typing import Any, Dict, List, cast

import torch

import torchfits as tf


def main():
    # Create and write a small VLA table (ragged: per-row different lengths)
    arrays = [
        torch.arange(0, 3, dtype=torch.float32),  # len 3
        torch.arange(0, 0, dtype=torch.float32),  # empty
        torch.arange(0, 5, dtype=torch.float32),  # len 5
        torch.tensor([42.0], dtype=torch.float32),  # len 1
    ]
    path = "vla_demo.fits"

    tf.write_variable_length_array(
        path, arrays, header={"EXTNAME": "VLA"}, overwrite=True
    )
    print(f"Wrote {path} with {len(arrays)} rows of ragged arrays.")

    # Read back the VLA table; the column is returned as list[Tensor]
    # read() returns (data, header) for tensor/table formats
    data, hdr = tf.read(path, hdu=1, format="tensor")
    data = cast(Dict[str, Any], data)
    vla = cast(List[torch.Tensor], data["ARRAY_DATA"])  # list[Tensor]
    print(f"Read table with {len(vla)} rows; first row tensor: {vla[0]}")

    # Convert to dense [rows, max_len] for batching
    padded, lengths = tf.pad_ragged(vla, pad_value=0.0)
    print("Padded shape:", padded.shape)
    print("Lengths:", lengths.tolist())

    # Example: masking/sorting with FitsTable still works for list-valued columns
    # Build a FitsTable with the ragged column and a numeric length column
    lens = torch.tensor([len(x) for x in vla], dtype=torch.int64)
    ft = tf.FitsTable({"ARRAY_DATA": vla, "LEN": lens})
    # Keep rows with length > 0
    ft_nz = ft[lens > 0]
    print(f"Non-empty rows: {len(ft_nz)}")
    # Sort by numeric length (supported)
    ft_sorted = ft.sort(column="LEN", descending=True)
    print("Sorted lengths:", cast(torch.Tensor, ft_sorted["LEN"]).tolist())


if __name__ == "__main__":
    main()
