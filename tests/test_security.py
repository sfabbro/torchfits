import pytest
import os
import torchfits


def test_security_cve_cfitsio_command_injection():
    """
    Test that filenames starting or ending with '|' are rejected to prevent
    CFITSIO command injection vulnerabilities.
    """
    # Filenames that should be rejected
    dangerous_filenames = [
        "| echo 'pwned'",
        " | ls",
        "valid.fits |",
        "valid.fits | ",
        "|/bin/sh -c 'touch /tmp/pwned'",
    ]

    for filename in dangerous_filenames:
        with pytest.raises(RuntimeError, match="Security Error"):
            torchfits.read(filename)


def test_valid_filenames_allowed():
    """Test that normal filenames are still allowed."""
    # We can't easily test success without a real file, but we can verify
    # it doesn't raise the Security Error.
    try:
        torchfits.read("nonexistent_file.fits")
    except RuntimeError as e:
        assert "Security Error" not in str(e)
    except FileNotFoundError:
        pass  # Expected
    except Exception:
        pass  # Other errors are fine, as long as not Security Error
