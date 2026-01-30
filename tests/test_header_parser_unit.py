
import unittest
import sys
import os

# Allow importing header_parser directly if torchfits dependencies are missing
try:
    from torchfits.header_parser import FastHeaderParser
except ImportError:
    # Add src/torchfits to path to import header_parser directly
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/torchfits")))
    from header_parser import FastHeaderParser

class TestHeaderParserUnit(unittest.TestCase):
    def test_basic_string(self):
        h = FastHeaderParser.parse_header_string("KEY     = 'value' / comment")
        self.assertEqual(h['KEY'], 'value')

    def test_escaped_quote(self):
        h = FastHeaderParser.parse_header_string("KEY     = 'val''ue' / comment")
        self.assertEqual(h['KEY'], "val'ue")

    def test_slash_in_string(self):
        h = FastHeaderParser.parse_header_string("KEY     = 'val / ue' / comment")
        self.assertEqual(h['KEY'], "val / ue")

    def test_numeric(self):
        h = FastHeaderParser.parse_header_string("KEY     = 123 / comment")
        self.assertEqual(h['KEY'], 123)
        self.assertIsInstance(h['KEY'], int)

    def test_numeric_signed(self):
        h = FastHeaderParser.parse_header_string("KEY     = -123 / comment")
        self.assertEqual(h['KEY'], -123)
        h = FastHeaderParser.parse_header_string("KEY     = +456 / comment")
        self.assertEqual(h['KEY'], 456)

    def test_float(self):
        h = FastHeaderParser.parse_header_string("KEY     = 123.456 / comment")
        self.assertAlmostEqual(h['KEY'], 123.456)
        self.assertIsInstance(h['KEY'], float)

    def test_logical(self):
        h = FastHeaderParser.parse_header_string("KEY     = T / comment")
        self.assertTrue(h['KEY'])
        self.assertIsInstance(h['KEY'], bool)

        h = FastHeaderParser.parse_header_string("KEY     = F / comment")
        self.assertFalse(h['KEY'])

    def test_unclosed_string(self):
        # This falls back to manual loop if regex fails
        h = FastHeaderParser.parse_header_string("KEY     = 'val")
        self.assertEqual(h['KEY'], "val")

    def test_garbage_after_string(self):
        # FITS allows this, usually parser ignores it
        h = FastHeaderParser.parse_header_string("KEY     = 'val' garbage")
        self.assertEqual(h['KEY'], "val")

    def test_find_comment_separator_optimization(self):
        # Test the fast path added
        # Case without quotes
        idx = FastHeaderParser._find_comment_separator("123 / comment")
        self.assertEqual(idx, 4)

        # Case with quotes
        idx = FastHeaderParser._find_comment_separator("'val' / comment")
        self.assertEqual(idx, 6)

        # Case with slash inside quotes
        idx = FastHeaderParser._find_comment_separator("'val / ue' / comment")
        self.assertEqual(idx, 11)

    def test_complex_header_string(self):
        # Multiple cards
        header_str = (
            "SIMPLE  =                    T / file does conform to FITS standard             "
            "BITPIX  =                  -32 / number of bits per data pixel                  "
            "NAXIS   =                    2 / number of data axes                            "
            "NAXIS1  =                  100 / length of data axis 1                          "
            "NAXIS2  =                  100 / length of data axis 2                          "
            "EXTEND  =                    T / FITS dataset may contain extensions            "
            "COMMENT   FITS (Flexible Image Transport System) format is defined in 'Astronomy"
            "COMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"
            "END                                                                             "
        )
        h = FastHeaderParser.parse_header_string(header_str)
        self.assertTrue(h['SIMPLE'])
        self.assertEqual(h['BITPIX'], -32)
        self.assertEqual(h['NAXIS'], 2)
        self.assertEqual(h['NAXIS1'], 100)

if __name__ == '__main__':
    unittest.main()
