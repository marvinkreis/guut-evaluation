from string_utils.manipulation import compress

def test_compress_empty_string_mutant_killing():
    """
    Test the compress function with an empty input string. The baseline should raise a ValueError indicating that
    the input string cannot be empty, while the mutant will not raise this exception due to the incorrect condition check.
    """
    try:
        compress('')
        # If we reach this line, the mutant did not raise the expected exception
        assert False, "Expected ValueError for empty input, but none was raised."
    except ValueError as e:
        # Check the message of the raised exception
        assert str(e) == 'Input string cannot be empty', f"Unexpected error message: {e}"