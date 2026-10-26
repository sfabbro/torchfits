import inspect

import numpy as np
import pytest

import torchfits
import torchfits.hdu as hdu
import torchfits.table as table
import torchfits.where as where


def test_hdu_and_table_public_surfaces_are_importable():
    assert hdu.DataView is not None
    assert callable(table.read)
    assert callable(table.write)
    assert callable(table.clear_cache)
    assert torchfits.hdu is hdu
    assert "hdu" in torchfits.__all__
    assert "read_fast" not in torchfits.__all__
    assert "SpectralBinning" not in torchfits.__all__
    assert "can_use_mmap_row_path_for_full_read" not in table.__all__
    assert "TableDataAccessor" not in hdu.__all__
    assert "where_identifier_re" not in where.__all__
    import torchfits.io as io

    assert not hasattr(io, "read_fast")
    assert "read_fast" not in io.__all__


def test_table_destination_readers_are_public(tmp_path):
    assert "read_arrow" in table.__all__
    assert "read_torch" in table.__all__
    assert table.read_arrow is table.read
    assert callable(table.read_torch)
    assert callable(torchfits.read_header)

    path = tmp_path / "boundary_table.fits"
    torchfits.table.write(
        str(path),
        {"ID": np.array([1, 2, 3], dtype=np.int64)},
        overwrite=True,
    )
    via_read = table.read(str(path), hdu=1)
    via_arrow = table.read_arrow(str(path), hdu=1)
    assert via_read.num_rows == via_arrow.num_rows == 3
    assert via_read.column("ID").to_pylist() == via_arrow.column("ID").to_pylist()


def test_torch_frame_is_not_part_of_the_fits_hdu_surface():
    assert not hasattr(hdu, "TensorFrame")


def test_removed_cache_knobs_are_gone():
    """1.3.0 removed the deprecated no-op cache knobs instead of keeping shims."""
    import torchfits.io as io
    from torchfits._io_engine.options import ReadOptions

    assert not hasattr(torchfits.cache, "configure_cache")
    for name in (
        "CacheConfig",
        "CacheManager",
        "configure_for_environment",
        "get_cache_manager",
        "get_optimal_cache_config",
        "optimize_for_dataset",
        "clear",
        "stats",
    ):
        assert not hasattr(torchfits.cache, name)
    with pytest.raises(TypeError):
        ReadOptions(handle_cache_capacity=16)
    with pytest.raises(TypeError):
        table.read_torch("unused.fits", handle_cache_capacity=16)
    with pytest.raises(TypeError):
        io.clear_file_cache(handles=False)
    with pytest.raises(TypeError):
        hdu.TableHDU({}, col_stats={})


def test_cache_policy_flags_match_clear_file_cache():
    """Every policy must be callable against ``clear_file_cache``."""
    import torchfits.io as io

    accepted = set(inspect.signature(io.clear_file_cache).parameters) - {"cpp_module"}
    for name in (
        "fits_image_data",
        "fits_table_data",
        "fits_header_metadata",
        "fits_header_hdu_metadata",
        "all",
    ):
        policy = io.cache_subsystem_policy(name)
        assert set(policy) == accepted
        io.clear_file_cache(**policy)


def test_table_hdu_header_is_keyword_only(tmp_path):
    """A positional header must raise, and the header must survive a write."""
    header = hdu.Header({"EXTNAME": "CAT", "NAXIS2": 1})
    with pytest.raises(TypeError):
        hdu.TableHDU({"A": np.arange(1)}, header)

    params = inspect.signature(hdu.TableHDU.__init__).parameters
    for name in ("header", "source_path", "source_hdu"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY

    table_hdu = hdu.TableHDU({"A": np.arange(1)}, header=header)
    path = tmp_path / "cat.fits"
    table_hdu.to_fits(str(path))
    assert torchfits.read_header(str(path), hdu=1)["EXTNAME"] == "CAT"


def test_root_functions_preserve_real_signatures():
    import torchfits.io

    assert inspect.signature(torchfits.read) == inspect.signature(torchfits.io.read)
    assert torchfits.read is torchfits.io.read


def test_cpp_public_surface_is_explicit_and_resolves():
    """The raw binding surface is private. There is no torchfits.cpp alias."""
    import importlib

    import torchfits._cpp as cpp

    assert len(cpp.__all__) == len(set(cpp.__all__))
    assert all(hasattr(cpp, name) for name in cpp.__all__)
    assert "read_header_dict" in cpp.__all__
    assert "resolve_hdu_name_cached" in cpp.__all__
    assert "compute_stats" not in cpp.__all__

    for gone in ("configure_cache", "get_cache_size", "clear_file_cache"):
        assert gone not in cpp.__all__

    assert "cpp" not in torchfits.__all__
    with pytest.raises(AttributeError):
        torchfits.cpp  # noqa: B018
    # A stale editable install still points at the deleted file.
    with pytest.raises((ModuleNotFoundError, FileNotFoundError)):
        importlib.import_module("torchfits.cpp")


def test_cpp_getattr_rejects_undocumented_native_names():
    import torchfits._C as native
    import torchfits._cpp as cpp

    assert hasattr(native, "HAS_BZIP2")
    with pytest.raises(AttributeError, match="HAS_BZIP2"):
        getattr(cpp, "HAS_BZIP2")
    assert "HAS_BZIP2" not in dir(cpp)


def test_where_public_surface_matches_table_predicate_semantics():
    ast = where.parse_where_expression("A > 1 AND B IS NOT NULL")
    mask = where.evaluate_where(
        ast, {"A": np.array([0, 2, 3]), "B": np.array([1, None, 4], dtype=object)}
    )
    np.testing.assert_array_equal(mask, np.array([False, False, True]))
    assert torchfits.where.parse_where_literal("42") == 42
    assert where.where_columns_from_ast(ast) == ["A", "B"]


def test_removed_root_aliases_are_not_public():
    removed = (
        "read_table",
        "stream_table",
        "read_table_rows",
        "get_header",
        "get_batch_info",
    )
    for name in removed:
        assert name not in torchfits.__all__
        with pytest.raises(AttributeError, match=name):
            getattr(torchfits, name)


def test_transforms_public_surface_excludes_removed_spectral_names():
    import torchfits.transforms as transforms

    removed = (
        "ContinuumNormalize",
        "ContinuumRemoval",
        "DopplerShift",
        "SpectralBinning",
        "BandMath",
        "PhaseFold",
        "SavitzkyGolayFilter",
        "RunningPercentile",
        "UpperEnvelopeContinuum",
        "WaveletDecompose",
        "AsymmetricLeastSquares",
        "AlphaShapeContinuum",
    )
    for name in removed:
        assert name not in transforms.__all__
        with pytest.raises(AttributeError, match=name):
            getattr(transforms, name)

    assert "AsModule" in transforms.__all__
    assert "as_module" in transforms.__all__
    assert callable(transforms.as_module)
