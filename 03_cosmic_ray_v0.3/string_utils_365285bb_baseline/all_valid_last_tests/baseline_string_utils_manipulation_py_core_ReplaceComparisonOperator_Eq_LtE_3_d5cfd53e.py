from string_utils.manipulation import compress

def test__compress_empty_string():
    # Test that compress raises ValueError for an empty string
    try:
        compress('')
    except ValueError as e:
        assert str(e) == 'Input string cannot be empty'  # This should pass for the correct implementation
    else:
        assert False, "Expected ValueError for empty string was not raised."

    # Test a valid input case to make sure the function works
    valid_string = "This is a test."
    compressed_string = compress(valid_string)
    assert compressed_string is not None, "Compression should return a non-null result for valid input."