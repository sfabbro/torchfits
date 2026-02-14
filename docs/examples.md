# TorchFits Examples

This directory contains examples of common `torchfits` workflows.

## Core Image/Table Scripts

| Example | Description |
|---------|-------------|
| `example_image.py` | Basic image reading and header access. |
| `example_table.py` | Basic table I/O operations (read, write, slice). |
| `example_table_interop.py` | Arrow/Pandas interop for table data. |
| `example_polars.py` | Polars-oriented table workflows. |

## WCS

| Example | Description |
|---------|-------------|
| `example_wcs_transform.py` | Header + coordinate transform examples for 1D/2D/3D WCS. |

## Image Handling

| Example | Description |
|---------|-------------|
| `example_image_cutouts.py` | Efficiently extracting sub-regions. |
| `example_image_cube.py` | Handling >2D data cubes. |
| `example_image_mef.py` | Working with Multi-Extension FITS (MEF) files. |
| `example_image_dataset.py` | Custom PyTorch Dataset for FITS images. |

## ML/Workflow

| Example | Description |
|---------|-------------|
| `example_ml_pipeline.py` | End-to-end ML pipeline on FITS data. |
| `example_phase2_features.py` | Feature-oriented workflow examples. |
| `example_table_recipes.py` | Practical table mutation/scan recipes. |

## Running Examples

All examples are executable Python scripts:

```bash
pixi run python examples/example_image.py
```
