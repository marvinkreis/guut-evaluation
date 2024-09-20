from string_utils.manipulation import compress

def test__compress_with_zero_level():
    """The mutant incorrectly restricts the compression level to a minimum of 1."""
    input_string = "This is a test string."
    output = compress(input_string, compression_level=0)
    
    # As output is expected to be a non-empty string in valid scenarios
    assert len(output) > 0, "compress should return a valid compressed string"