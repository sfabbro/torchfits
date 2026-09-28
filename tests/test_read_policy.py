"""Read-policy unit tests."""

from torchfits._table_engine import (
    WhereStrategy,
    choose_where_read_plan,
    should_skip_cpp_for_where,
    validate_table_backend,
)


def test_should_skip_cpp_for_where():
    assert should_skip_cpp_for_where("auto", "MAG < 20") is True
    assert should_skip_cpp_for_where("auto", None) is False
    assert should_skip_cpp_for_where("cpp", "MAG < 20") is False


def test_choose_where_read_plan_auto_mmap_uses_pushdown():
    """auto + mmap=True ⇒ documented native mmap-scan pushdown."""
    plan = choose_where_read_plan(
        header={},
        header_ok=True,
        columns=None,
        backend="auto",
        n_rows=1_000,
        mmap=True,
    )
    assert plan.strategy == WhereStrategy.CPP_PUSHDOWN
    assert plan.unfiltered_backend == "cpp"
    assert plan.cpp_pushdown_safe is True


def test_choose_where_read_plan_auto_buffered_uses_arrow_filter():
    """auto + mmap=False ⇒ read then tensor/Arrow filter (no size forks)."""
    for n_rows in (1_000, 100_000):
        plan = choose_where_read_plan(
            header={},
            header_ok=True,
            columns=None,
            backend="auto",
            n_rows=n_rows,
            mmap=False,
        )
        assert plan.strategy == WhereStrategy.ARROW_FILTER
        assert plan.unfiltered_backend == "cpp"


def test_choose_where_read_plan_cpp_uses_pushdown():
    plan = choose_where_read_plan(
        header={},
        header_ok=True,
        columns=None,
        backend="cpp",
        n_rows=10_000,
        mmap=True,
    )
    assert plan.strategy == WhereStrategy.CPP_PUSHDOWN


def test_cpp_numpy_backend_rejected():
    """The legacy 'cpp_numpy' backend alias was removed in 0.8.0."""
    import pytest

    with pytest.raises(ValueError, match="backend must be one of"):
        validate_table_backend("cpp_numpy")


def test_unhashable_backend_reports_the_public_error():
    """Membership in a frozenset hashes its argument, so an unhashable
    backend used to leak ``TypeError: unhashable type`` from the public
    table.read/scan entry points instead of the documented ValueError -- and
    only for unhashable types, since a tuple or bytes already gave ValueError.
    """
    import pytest

    for bad in (["cpp"], {"a": 1}, {"cpp"}, bytearray(b"cpp"), 1 + 2j):
        with pytest.raises(ValueError, match="backend must be one of"):
            validate_table_backend(bad)


def test_unhashable_backend_is_rejected_through_the_public_api(tmp_path):
    """The leak was reachable from table.read/scan, not just the helper."""
    import numpy as np
    import pytest
    from astropy.io import fits

    import torchfits

    path = str(tmp_path / "t.fits")
    cols = [fits.Column(name="X", format="E", array=np.arange(4, dtype=np.float32))]
    fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols, nrows=4)]
    ).writeto(path, overwrite=True)

    with pytest.raises(ValueError, match="backend must be one of"):
        torchfits.table.read(path, hdu=1, backend=["cpp"])
    with pytest.raises(ValueError, match="backend must be one of"):
        list(torchfits.table.scan(path, hdu=1, backend={"a": 1}))
