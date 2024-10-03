from string_utils.manipulation import compress

def test_compress_invalid_compression_level():
    """
    Test the compress function with a compression level of 10,
    which is above the maximum acceptable range. The baseline
    should raise a ValueError, while the mutant should raise a
    zlib.error indicating a bad compression level.
    """
    # Testing the baseline
    try:
        compress('Test string for compression', compression_level=10)
        raise AssertionError("Expected ValueError not raised in baseline.")
    except ValueError as e:
        print(f"Baseline correctly raised ValueError: {e}")  # Expect baseline error

    # Testing the mutant
    try:
        output = compress('Test string for compression', compression_level=10)
        print(f"Output from mutant (compression_level=10): {output}")  # Expect a mutant behavior
    except Exception as e:
        print(f"Mutant raised an exception: {e}")  # Expect an exception that indicates mutant behavior