from string_utils.manipulation import compress

def test__compress_with_invalid_level():
    """
    This test verifies the behavior of the compress function when the compression level exceeds the allowed maximum.
    The input is a standard string and should result in a successful compression with the baseline but should raise
    a ValueError in the mutant due to an invalid default compression level of 10.
    """
    input_string = "This is a sample string for testing compression behavior."
    # Expecting it to pass on baseline, but fail on mutant
    output = compress(input_string)
    assert isinstance(output, str)  # Should be a non-empty string in baseline