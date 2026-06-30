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
        "!| echo 'pwned'",
        "!! | ls",
        "! !| id",
    ]

    for filename in dangerous_filenames:
        with pytest.raises(RuntimeError, match="Security Error"):
            torchfits.read(filename)


def test_forced_overwrite_prefix_allowed():
    """Leading '!' is valid CFITSIO overwrite syntax and must not bypass pipe checks."""
    try:
        torchfits.read("!nonexistent_file.fits")
    except RuntimeError as e:
        assert "Security Error" not in str(e)
    except FileNotFoundError:
        pass
    except Exception:
        pass


def test_header_large_dict_construction_fast():
    """Regression: Header(dict) must stay O(N), not O(N^2) (PR #172)."""
    import time

    from torchfits.hdu import Header

    d = {f"KEY{i}": i for i in range(2000)}
    t0 = time.perf_counter()
    h = Header(d)
    elapsed = time.perf_counter() - t0
    assert len(h) == 2000
    assert elapsed < 0.5, f"Header(2000) took {elapsed:.3f}s; expected sub-second"


def test_valid_filenames_allowed():
    """Test that normal filenames are still allowed."""
    try:
        torchfits.read("nonexistent_file.fits")
    except RuntimeError as e:
        assert "Security Error" not in str(e)
    except FileNotFoundError:
        pass
    except Exception:
        pass
