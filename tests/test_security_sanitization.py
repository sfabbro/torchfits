import os
import pytest
import torchfits

def create_malicious_fits(filename):
    # Create a minimal FITS file manually
    header_content = [
        b"SIMPLE  =                    T / Standard FITS format                           ",
        b"BITPIX  =                    8 / Character or unsigned binary integer           ",
        b"NAXIS   =                    0 / No image data                                  ",
        b"EXTEND  =                    T / There may be standard extensions               ",
        b"COMMENT   Malicious comment \x07\x1b[31mRED\x1b[0m                                     ",
        b"BAD\x07KEY = 'BadValue'           / Key with control char                          ",
        b"END                                                                             "
    ]

    # Pad to 80 chars
    header_block = b""
    for line in header_content:
        if len(line) < 80:
            line += b" " * (80 - len(line))
        header_block += line

    # Pad to 2880 bytes
    if len(header_block) < 2880:
        header_block += b" " * (2880 - len(header_block))

    with open(filename, "wb") as f:
        f.write(header_block)

def test_header_sanitization(tmp_path):
    """Test that FITS headers are sanitized of control characters."""
    filename = str(tmp_path / "test_malicious.fits")
    create_malicious_fits(filename)

    # Read header
    # Note: Depending on how torchfits is built (fast_header=True/False),
    # it might use different code paths. We should test default behavior.
    try:
        header = torchfits.get_header(filename)
    except Exception as e:
        pytest.fail(f"Failed to read header: {e}")

    # Check comments
    comments = header.get_comment()
    for c in comments:
        # Check for control characters (ASCII < 32 except maybe newline/tab if allowed, but usually not in FITS)
        # We specifically targeted \x07 (bell) and \x1b (escape)
        assert "\x07" not in c, f"Control character \\x07 found in comment: {repr(c)}"
        assert "\x1b" not in c, f"Control character \\x1b found in comment: {repr(c)}"
        # Check that printable text remains
        assert "Malicious comment" in c, "Valid text removed from comment"
        assert "[31mRED[0m" in c, "Safe parts of escape sequence removed or corrupted"

    # Check keys
    keys = list(header.keys())
    # Note: Header keys are usually uppercase in dict if read via cfitsio which normalizes them?
    # Or strict? Our manual file has "BAD\x07KEY".
    # If key was sanitized, it should be "BADKEY".

    found_sanitized_key = False
    for k in keys:
        assert "\x07" not in k, f"Control character \\x07 found in key: {repr(k)}"
        assert "\x1b" not in k, f"Control character \\x1b found in key: {repr(k)}"
        if k == "BADKEY":
            found_sanitized_key = True

    assert found_sanitized_key, "Sanitized key 'BADKEY' not found in header"

if __name__ == "__main__":
    # Allow running directly
    import sys
    sys.exit(pytest.main([__file__]))
