from string_utils.manipulation import compress

def test_compress_mutant_killing():
    """
    Test the compress function using a compression_level of 0.
    The baseline should return a compressed string without error,
    while the mutant should raise a ValueError due to the level being too low.
    """
    input_string = "This string is intended for compression testing."
    output = compress(input_string, compression_level=0)  # Expecting this to work in the baseline
    assert isinstance(output, str), f"Expected a string output, got {type(output)}"