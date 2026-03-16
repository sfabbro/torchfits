import logging
import pytest
from torchfits.logging import (
    logger,
    set_log_level,
    log_errors,
    log_performance,
    log_fits_error,
    log_memory_usage,
    log_performance_warning,
)


def test_set_log_level():
    """Test that setting log levels works correctly."""
    # Test valid levels
    set_log_level("DEBUG")
    assert logger.level == logging.DEBUG

    set_log_level("WARNING")
    assert logger.level == logging.WARNING

    # Test case insensitivity
    set_log_level("error")
    assert logger.level == logging.ERROR

    # Test invalid level defaults to INFO
    set_log_level("INVALID_LEVEL")
    assert logger.level == logging.INFO


def test_log_errors_decorator(caplog):
    """Test that log_errors decorator logs exceptions and re-raises them."""
    @log_errors
    def failing_function():
        raise ValueError("Test error")

    @log_errors
    def passing_function():
        return "success"

    # Test passing function
    assert passing_function() == "success"
    assert not caplog.records

    # Test failing function
    with pytest.raises(ValueError, match="Test error"):
        failing_function()

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.ERROR
    assert "Error in failing_function: Test error" in record.message


def test_log_performance_decorator(caplog):
    """Test that log_performance decorator logs timing info."""
    # Ensure debug logging is captured
    set_log_level("DEBUG")
    caplog.set_level(logging.DEBUG, logger="torchfits")

    @log_performance
    def quick_function():
        return "done"

    @log_performance
    def failing_function():
        raise RuntimeError("Fail!")

    # Test successful function
    caplog.clear()
    assert quick_function() == "done"
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.DEBUG
    assert "quick_function completed in" in caplog.records[0].message

    # Test failing function
    caplog.clear()
    with pytest.raises(RuntimeError, match="Fail!"):
        failing_function()

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    assert "failing_function failed after" in caplog.records[0].message
    assert "Fail!" in caplog.records[0].message


def test_log_fits_error(caplog):
    """Test logging CFITSIO errors."""
    caplog.set_level(logging.ERROR, logger="torchfits")

    # Without details
    log_fits_error("read_file", 104)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    assert "CFITSIO error in read_file: status=104" in caplog.records[0].message

    caplog.clear()

    # With details
    log_fits_error("write_file", 105, "Could not open file")
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    assert "CFITSIO error in write_file: status=105, details=Could not open file" in caplog.records[0].message


def test_log_memory_usage(caplog):
    """Test memory usage logging."""
    set_log_level("DEBUG")
    caplog.set_level(logging.DEBUG, logger="torchfits")

    log_memory_usage("loading tensor", 123.456)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.DEBUG
    assert "Memory usage in loading tensor: 123.46 MB" in caplog.records[0].message


def test_log_performance_warning(caplog):
    """Test performance warning logging."""
    caplog.set_level(logging.WARNING, logger="torchfits")

    # Should not log (under threshold)
    log_performance_warning("fast_op", 500.0, threshold_ms=1000.0)
    assert len(caplog.records) == 0

    # Should log (over threshold)
    log_performance_warning("slow_op", 1500.0, threshold_ms=1000.0)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "Slow operation detected: slow_op took 1500.00ms (threshold: 1000.0ms)" in caplog.records[0].message

    # Test default threshold (1000ms)
    caplog.clear()
    log_performance_warning("default_slow_op", 1200.0)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "Slow operation detected: default_slow_op took 1200.00ms (threshold: 1000ms)" in caplog.records[0].message
