import pytest
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


def test_security_safe_eval_restrictions():
    """Test that _safe_eval restricts numpy attributes and functions."""
    from torchfits.hdu import _safe_eval
    import numpy as np

    # Should be allowed
    assert _safe_eval("np.sqrt(4)", {}, np) == 2.0
    assert _safe_eval("np.pi", {}, np) == np.pi
    assert _safe_eval("np.where(np.array([True, False]), 1, 0)", {}, np)[0] == 1

    # Should be blocked
    dangerous_cases = [
        "np.save('test.npy', [1])",
        "np.load('test.npy')",
        "np.random.rand()",
        "np.__class__",
        "np.core.multiarray",
    ]

    for case in dangerous_cases:
        with pytest.raises((ValueError, AttributeError)):
            _safe_eval(case, {}, np)
