# Mock out torchfits.cpp since we only care about header_parser.py testing right now

from torchfits.header_parser import benchmark_header_parsing


def test_benchmark_header_parsing():
    """Test the benchmark_header_parsing function."""
    # Create a simple valid FITS header string
    header_cards = [
        "SIMPLE  =                    T / file does conform to FITS standard             ",
        "BITPIX  =                    8 / number of bits per data pixel                  ",
        "NAXIS   =                    0 / number of data axes                            ",
        "EXTEND  =                    T / FITS dataset may contain extensions            ",
        "END                                                                             ",
    ]
    # Pad to exactly 80 characters per card
    header_cards = [card.ljust(80) for card in header_cards]
    header_string = "".join(header_cards)

    # Run benchmark
    num_iterations = 10
    metrics = benchmark_header_parsing(header_string, num_iterations=num_iterations)

    # Check metrics dictionary keys and types
    assert isinstance(metrics, dict)

    expected_keys = [
        "avg_parse_time_ms",
        "throughput_headers_per_sec",
        "total_time_s",
        "header_size_bytes",
        "num_iterations",
    ]

    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], (float, int))

    # Check specific values
    assert metrics["num_iterations"] == num_iterations
    assert metrics["header_size_bytes"] == len(header_string)
    assert metrics["avg_parse_time_ms"] >= 0
    assert metrics["throughput_headers_per_sec"] >= 0
    assert metrics["total_time_s"] >= 0
