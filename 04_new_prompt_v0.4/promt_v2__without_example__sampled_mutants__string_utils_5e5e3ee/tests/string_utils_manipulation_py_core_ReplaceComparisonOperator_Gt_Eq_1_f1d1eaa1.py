from string_utils.manipulation import compress

def test__compress_with_valid_level():
    """
    Test the compress function with valid compression level (9).
    The baseline should pass without raising any errors, while the mutant should raise a ValueError.
    """
    input_string = 'sample text for compression'

    # For baseline (which allows 9)
    output = compress(input_string, compression_level=9)
    assert isinstance(output, str)  # Ensure it returns a string for the baseline

    # Now we explicitly attempt to call it with a valid level (9) and will rely on the mutant detection
    try:
        compress(input_string, compression_level=9)
    except ValueError as e:
        print(f"ValueError raised as expected: {e}")