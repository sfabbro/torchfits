"""
Table-specific benchmarks for torchfits TableHDU operations.

Benchmarks table reading, filtering, and pytorch-frame integration
across different table sizes and column types.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import fitsio
from astropy.io import fits as astropy_fits
from astropy.table import Table

import torchfits

# Table test configurations
TABLE_SIZES = [1000, 10000, 100000, 1000000]
COLUMN_CONFIGS = [
    {"ncols": 5, "types": ["f4"] * 5},
    {"ncols": 20, "types": ["f4"] * 10 + ["i4"] * 10},
    {"ncols": 50, "types": ["f4"] * 25 + ["i4"] * 15 + ["U10"] * 10},
]


class TableBenchmarkData:
    """Generate table test data."""

    @staticmethod
    def create_table_data(nrows, ncols=10, column_types=None):
        """Create table data with specified types."""
        if column_types is None:
            column_types = ["f4"] * ncols

        data = {}
        for i, dtype in enumerate(column_types[:ncols]):
            col_name = f"col_{i}"

            if dtype.startswith("f"):  # float
                data[col_name] = np.random.randn(nrows).astype(dtype)
            elif dtype.startswith("i"):  # int
                data[col_name] = np.random.randint(0, 1000, nrows).astype(dtype)
            elif dtype.startswith("U"):  # string
                str_len = int(dtype[1:]) if len(dtype) > 1 else 10
                data[col_name] = [f"str_{j:0{str_len - 4}d}" for j in range(nrows)]
            else:
                data[col_name] = np.random.randn(nrows).astype("f4")

        return data

    @staticmethod
    def write_table_fits(data, filename):
        """Write table FITS file using astropy."""

        # Convert to astropy table
        table = Table(data)
        table.write(filename, format="fits", overwrite=True)
        return filename


class TableBenchmarkSuite:
    """Table-specific benchmark suite."""

    def __init__(self, table_sizes=None, column_configs=None):
        self.results = {}
        self.temp_dir = tempfile.mkdtemp()
        self.table_sizes = (
            list(table_sizes) if table_sizes is not None else list(TABLE_SIZES)
        )
        self.column_configs = (
            list(column_configs) if column_configs is not None else list(COLUMN_CONFIGS)
        )

    def benchmark_table_reading(self):
        """Benchmark table reading operations."""
        print("=== Table Reading Benchmarks ===")

        for nrows in self.table_sizes:
            for config in self.column_configs:
                ncols = config["ncols"]
                types = config["types"]

                print(f"\nTable: {nrows} rows, {ncols} columns")

                # Create test data
                data = TableBenchmarkData.create_table_data(nrows, ncols, types)
                filename = Path(self.temp_dir) / f"table_{nrows}_{ncols}.fits"

                if TableBenchmarkData.write_table_fits(data, filename):
                    self._benchmark_table_read_operations(
                        filename, f"table_{nrows}_{ncols}"
                    )

    def benchmark_table_operations(self):
        """Benchmark table query operations."""
        print("\n=== Table Query Benchmarks ===")

        # Create medium-sized test table
        nrows = 100000
        data = TableBenchmarkData.create_table_data(nrows, 10)
        filename = Path(self.temp_dir) / "query_test.fits"

        if not TableBenchmarkData.write_table_fits(data, filename):
            print("Skipping table operations (no astropy)")
            return

        print(f"\nQuery operations on {nrows} row table:")

        try:
            # Test lazy operations (should be fast)
            with torchfits.open(str(filename)) as hdul:
                if len(hdul) > 1:  # Table in extension
                    table = hdul[1]
                else:
                    print("No table HDU found")
                    return

                # Selection benchmark (view/projection)
                start_time = time.perf_counter()
                _ = table.select(["col_0", "col_1", "col_2"])
                select_time = time.perf_counter() - start_time
                print(f"  Column selection: {select_time:.6f}s (view)")

                # Filter benchmark (out-of-core Arrow batch)
                start_time = time.perf_counter()
                try:
                    import pyarrow.dataset as ds

                    scanner = torchfits.table.scanner(
                        str(filename),
                        columns=["col_0", "col_1"],
                        filter=ds.field("col_0") > 0,
                        batch_size=65536,
                        decode_bytes=False,
                    )
                    _ = next(scanner.to_batches())
                    filter_time = time.perf_counter() - start_time
                    print(f"  Row filtering: {filter_time:.6f}s (arrow batch)")
                except Exception as e:
                    filter_time = time.perf_counter() - start_time
                    print(f"  Row filtering: FAILED ({e})")

                # Head benchmark (view)
                start_time = time.perf_counter()
                _ = table.head(1000)
                head_time = time.perf_counter() - start_time
                print(f"  Head operation: {head_time:.6f}s (view)")

                # Chained view ops
                start_time = time.perf_counter()
                chained = table.select(["col_0", "col_1"]).head(1000)
                chain_time = time.perf_counter() - start_time
                print(f"  Chained operations: {chain_time:.6f}s (view)")

                # Materialization benchmark (explicit and potentially expensive)
                start_time = time.perf_counter()
                chained.materialize()
                materialize_time = time.perf_counter() - start_time
                print(f"  Materialization: {materialize_time:.4f}s")

        except Exception as e:
            print(f"Table operations failed: {e}")

    def benchmark_column_access(self):
        """Benchmark individual column access."""
        print("\n=== Column Access Benchmarks ===")

        nrows = 50000
        data = TableBenchmarkData.create_table_data(nrows, 20)
        filename = Path(self.temp_dir) / "column_test.fits"

        if not TableBenchmarkData.write_table_fits(data, filename):
            print("Skipping column access (no astropy)")
            return

        print(f"\nColumn access on {nrows} row table:")

        try:
            with torchfits.open(str(filename)) as hdul:
                if len(hdul) > 1:
                    table = hdul[1]

                    # Get available column names
                    available_cols = list(getattr(table, "columns", []))
                    print(f"  Available columns: {available_cols[:5]}...")

                    if available_cols:
                        # Single column access using first available column
                        first_col = available_cols[0]
                        start_time = time.perf_counter()
                        table[first_col]
                        single_time = time.perf_counter() - start_time
                        print(f"  Single column: {single_time:.4f}s")

                        # Multiple column access
                        start_time = time.perf_counter()
                        table.to_tensor_dict()
                        multi_time = time.perf_counter() - start_time
                        print(f"  All columns: {multi_time:.4f}s")
                    else:
                        print("  No columns available")

        except Exception as e:
            print(f"Column access failed: {e}")

    def benchmark_streaming(self):
        """Benchmark table streaming operations."""
        print("\n=== Table Streaming Benchmarks ===")

        nrows = 100000
        data = TableBenchmarkData.create_table_data(nrows, 10)
        filename = Path(self.temp_dir) / "stream_test.fits"

        if not TableBenchmarkData.write_table_fits(data, filename):
            print("Skipping streaming (no astropy)")
            return

        batch_sizes = [1000, 5000, 10000]

        for batch_size in batch_sizes:
            print(f"\nStreaming with batch_size={batch_size}:")

            try:
                with torchfits.open(str(filename)) as hdul:
                    if len(hdul) > 1:
                        table = hdul[1]

                        start_time = time.perf_counter()
                        total_rows = 0

                        for batch in table.iter_rows(batch_size=batch_size):
                            if isinstance(batch, dict):
                                # Count rows in dict format
                                if batch:
                                    first_col = next(iter(batch.values()))
                                    if hasattr(first_col, "shape"):
                                        total_rows += first_col.shape[0]
                                    else:
                                        total_rows += (
                                            len(first_col)
                                            if hasattr(first_col, "__len__")
                                            else 1
                                        )
                                break  # Just test one batch for timing
                            else:
                                total_rows += (
                                    len(batch) if hasattr(batch, "__len__") else 1
                                )
                                break

                        stream_time = time.perf_counter() - start_time
                        throughput = total_rows / stream_time if stream_time > 0 else 0

                        print(f"  Time: {stream_time:.4f}s")
                        print(f"  Throughput: {throughput:.0f} rows/sec")

            except Exception as e:
                print(f"Streaming failed: {e}")

    def _benchmark_table_read_operations(self, filename, label):
        """Benchmark table reading with different libraries."""
        results = {}

        # torchfits
        try:
            start_time = time.perf_counter()
            with torchfits.open(str(filename)) as hdul:
                if len(hdul) > 1:
                    table = hdul[1]
                    # Just access the data to ensure it's loaded
                    _ = table.num_rows
            torchfits_time = time.perf_counter() - start_time
            results["torchfits"] = torchfits_time
            print(f"  torchfits: {torchfits_time:.4f}s")
        except Exception as e:
            print(f"  torchfits: FAILED ({e})")

        # astropy
        try:
            start_time = time.perf_counter()
            with astropy_fits.open(filename) as hdul:
                hdul[1].data
            astropy_time = time.perf_counter() - start_time
            results["astropy"] = astropy_time
            print(f"  astropy: {astropy_time:.4f}s")
        except Exception as e:
            print(f"  astropy: FAILED ({e})")

        # fitsio
        try:
            start_time = time.perf_counter()
            fitsio.read(str(filename), ext=1)
            fitsio_time = time.perf_counter() - start_time
            results["fitsio"] = fitsio_time
            print(f"  fitsio: {fitsio_time:.4f}s")
        except Exception as e:
            print(f"  fitsio: FAILED ({e})")

        self.results[label] = results

    def print_summary(self):
        """Print table benchmark summary."""
        print("\n" + "=" * 60)
        print("TABLE BENCHMARK SUMMARY")
        print("=" * 60)

        if not self.results:
            print("No table benchmark results available")
            return

        for test_name, results in self.results.items():
            print(f"\n{test_name}:")

            if "torchfits" in results and "astropy" in results:
                speedup = results["astropy"] / results["torchfits"]
                print(f"  torchfits vs astropy: {speedup:.2f}x")

            if "torchfits" in results and "fitsio" in results:
                speedup = results["fitsio"] / results["torchfits"]
                print(f"  torchfits vs fitsio: {speedup:.2f}x")

    def run_all_benchmarks(self):
        """Run complete table benchmark suite."""
        print("torchfits Table Benchmark Suite")
        print("=" * 40)

        self.benchmark_table_reading()
        self.benchmark_table_operations()
        self.benchmark_column_access()
        self.benchmark_streaming()
        self.print_summary()


def _quick_column_configs():
    return [
        {"ncols": 5, "types": ["f4"] * 5},
        {"ncols": 20, "types": ["f4"] * 10 + ["i4"] * 10},
    ]


def main():
    """Run table benchmark suite."""
    parser = argparse.ArgumentParser(description="Run table benchmark suite")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced table sizes/configs for fast iteration.",
    )
    args = parser.parse_args()

    table_sizes = TABLE_SIZES
    column_configs = COLUMN_CONFIGS
    if args.quick:
        table_sizes = [1000, 10000, 100000]
        column_configs = _quick_column_configs()

    suite = TableBenchmarkSuite(table_sizes=table_sizes, column_configs=column_configs)
    suite.run_all_benchmarks()


if __name__ == "__main__":
    main()
