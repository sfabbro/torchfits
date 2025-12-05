"""
Example: PyTorch-Frame Integration for Tabular Data

This example demonstrates how to use torchfits with pytorch-frame
for tabular deep learning on astronomical catalogs.

pytorch-frame provides a unified interface for tabular data with
automatic feature type inference and deep learning models.
"""

import tempfile

import numpy as np
import torch
import torchfits
from astropy.io import fits as astropy_fits
from astropy.table import Table

# Check if pytorch-frame is available
try:
    from torch_frame import stype

    HAS_TORCH_FRAME = True
except ImportError:
    print("pytorch-frame not installed. Install with: pip install pytorch-frame")
    HAS_TORCH_FRAME = False
    exit(1)


def create_sample_catalog():
    """Create a sample astronomical catalog FITS file."""
    print("Creating sample catalog...")

    # Simulate a source catalog with mixed data types
    n_sources = 10000

    catalog_data = {
        # Numerical features (astrometry, photometry)
        "ra": np.random.uniform(0, 360, n_sources).astype(np.float64),
        "dec": np.random.uniform(-90, 90, n_sources).astype(np.float64),
        "ra_err": np.abs(np.random.normal(0, 0.1, n_sources)).astype(np.float32),
        "dec_err": np.abs(np.random.normal(0, 0.1, n_sources)).astype(np.float32),
        "g_mag": np.random.uniform(15, 25, n_sources).astype(np.float32),
        "r_mag": np.random.uniform(15, 25, n_sources).astype(np.float32),
        "i_mag": np.random.uniform(15, 25, n_sources).astype(np.float32),
        "g_mag_err": np.abs(np.random.normal(0, 0.05, n_sources)).astype(np.float32),
        "r_mag_err": np.abs(np.random.normal(0, 0.05, n_sources)).astype(np.float32),
        "i_mag_err": np.abs(np.random.normal(0, 0.05, n_sources)).astype(np.float32),
        # Categorical features
        "source_type": np.random.randint(0, 3, n_sources).astype(
            np.int64
        ),  # 0=star, 1=galaxy, 2=qso
        "quality_flag": np.random.choice([True, False], n_sources),
        # Derived features
        "g_r_color": np.zeros(n_sources, dtype=np.float32),
        "r_i_color": np.zeros(n_sources, dtype=np.float32),
    }

    # Compute colors
    catalog_data["g_r_color"] = catalog_data["g_mag"] - catalog_data["r_mag"]
    catalog_data["r_i_color"] = catalog_data["r_mag"] - catalog_data["i_mag"]

    # Create FITS table
    table = Table(catalog_data)
    filename = tempfile.mktemp(suffix=".fits")
    table.write(filename, format="fits", overwrite=True)

    print(f"  Created catalog with {n_sources} sources")
    print(f"  File: {filename}")
    return filename


def example_basic_conversion():
    """Basic conversion from FITS to TensorFrame."""
    print("\n" + "=" * 60)
    print("Example 1: Basic FITS to TensorFrame Conversion")
    print("=" * 60)

    # Create sample catalog
    catalog_file = create_sample_catalog()

    # Method 1: Direct read as TensorFrame
    print("\nMethod 1: read_tensor_frame()")
    tf = torchfits.read_tensor_frame(catalog_file, hdu=1)

    print(f"  TensorFrame shape: {tf.num_rows} rows")
    print(f"  Numerical features: {tf.feat_dict[stype.numerical].shape}")
    print(f"  Categorical features: {tf.feat_dict[stype.categorical].shape}")
    print(f"\n  Numerical columns: {tf.col_names_dict[stype.numerical]}")
    print(f"  Categorical columns: {tf.col_names_dict[stype.categorical]}")

    # Method 2: Convert from dictionary
    print("\nMethod 2: read() + to_tensor_frame()")
    data, header = torchfits.read(catalog_file, hdu=1)
    tf2 = torchfits.to_tensor_frame(data)

    print(f"  TensorFrame shape: {tf2.num_rows} rows")
    print(f"  Matches direct read: {tf2.num_rows == tf.num_rows}")


def example_selective_loading():
    """Load only specific columns as TensorFrame."""
    print("\n" + "=" * 60)
    print("Example 2: Selective Column Loading")
    print("=" * 60)

    catalog_file = create_sample_catalog()

    # Select only photometry columns
    photo_columns = ["g_mag", "r_mag", "i_mag", "g_r_color", "r_i_color"]

    data, _ = torchfits.read(catalog_file, hdu=1, columns=photo_columns)
    tf = torchfits.to_tensor_frame(data)

    print(f"  Loaded {len(photo_columns)} photometric columns")
    print(f"  TensorFrame features: {tf.feat_dict[stype.numerical].shape}")
    print(f"  Column names: {tf.col_names_dict[stype.numerical]}")


def example_write_tensorframe():
    """Write TensorFrame back to FITS."""
    print("\n" + "=" * 60)
    print("Example 3: Writing TensorFrame to FITS")
    print("=" * 60)

    catalog_file = create_sample_catalog()

    # Read as TensorFrame
    tf = torchfits.read_tensor_frame(catalog_file, hdu=1)
    print(f"  Original TensorFrame: {tf.num_rows} rows")

    # Write back to FITS
    output_file = tempfile.mktemp(suffix=".fits")
    torchfits.write_tensor_frame(output_file, tf, overwrite=True)
    print(f"  Written to: {output_file}")

    # Verify by reading back
    tf_read = torchfits.read_tensor_frame(output_file, hdu=1)
    print(
        f"  Verified: {tf_read.num_rows} rows, {len(tf_read.col_names_dict[stype.numerical])} numerical cols"
    )


def example_ml_workflow():
    """Complete ML workflow with TensorFrame."""
    print("\n" + "=" * 60)
    print("Example 4: Machine Learning Workflow")
    print("=" * 60)

    catalog_file = create_sample_catalog()

    # Load data
    tf = torchfits.read_tensor_frame(catalog_file, hdu=1)
    print(f"  Loaded {tf.num_rows} sources")

    # Split features and target
    # For this example, we'll use source_type as the target
    # In pytorch-frame, categorical columns are separate
    cat_idx = tf.col_names_dict[stype.categorical].index("source_type")
    target = tf.feat_dict[stype.categorical][:, cat_idx]

    print(f"\n  Target distribution (source_type):")
    for i in range(3):
        count = (target == i).sum().item()
        print(f"    Type {i}: {count} ({100*count/tf.num_rows:.1f}%)")

    # Create simple splits
    n_train = int(0.7 * tf.num_rows)
    n_val = int(0.15 * tf.num_rows)

    print(f"\n  Data splits:")
    print(f"    Train: {n_train}")
    print(f"    Val: {n_val}")
    print(f"    Test: {tf.num_rows - n_train - n_val}")

    # Example: Access numerical features for training
    X = tf.feat_dict[stype.numerical]
    print(f"\n  Feature matrix shape: {X.shape}")
    print(f"  Features: {tf.col_names_dict[stype.numerical]}")

    # You can now use this data with any pytorch-frame model
    # See pytorch-frame documentation for model examples


def main():
    """Run all examples."""
    if not HAS_TORCH_FRAME:
        return

    print("PyTorch-Frame Integration Examples")
    print("=" * 60)

    example_basic_conversion()
    example_selective_loading()
    example_write_tensorframe()
    example_ml_workflow()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
