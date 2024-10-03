from string_utils.manipulation import compress

def test_compress_mutant_killing():
    """
    Test the compress function using default values for encoding and compression_level.
    The mutant will raise a ValueError due to an invalid compression_level of 10,
    while the baseline will execute successfully with the default compression_level of 9.
    """
    output = compress("This is a test string.")
    assert output is not None, "Expected output to be not None"