import pytest
import torchfits

def test_filename_injection_protection():
    """Test that filenames starting or ending with | are rejected to prevent command injection."""

    # Test cases that should fail
    dangerous_filenames = [
        "|echo vulnerable",
        "echo vulnerable|",
        " | echo vulnerable",  # Spaces are trimmed
        "echo vulnerable | ",
    ]

    for filename in dangerous_filenames:
        # Expect RuntimeError with specific message
        # Note: torchfits.read wraps exceptions, so we check the string representation
        try:
            torchfits.read(filename)
        except ImportError:
            pytest.skip("C++ extension not installed")
        except RuntimeError as e:
            assert "Security Error" in str(e) or "Security Error" in str(e.__cause__), \
                   f"Expected Security Error for filename '{filename}', but got: {e}"
        except Exception as e:
            pytest.fail(f"Unexpected exception type {type(e)}: {e}")

def test_read_image_fast_injection_protection():
    """Test read_image_fast specifically if exposed."""
    try:
        # Attempt to import the binding directly if available
        # It might be under torchfits.cpp or not exposed directly in python package
        import torchfits.cpp as cpp
        if not hasattr(cpp, 'read_image_fast'):
            pytest.skip("read_image_fast not exposed in python bindings")

        read_image_fast = cpp.read_image_fast
    except ImportError:
        pytest.skip("C++ extension not installed")

    filename = "|echo vulnerable"
    with pytest.raises(RuntimeError, match="Security Error"):
        read_image_fast(filename, 0, True)
