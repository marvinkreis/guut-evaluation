from string_utils.manipulation import compress

def test__compress_empty_string():
    # Test case to detect the mutant that changes the empty string check
    try:
        compress('', compression_level=9)
        assert False, "Expected ValueError for empty string input, but no exception was raised."
    except ValueError as e:
        assert str(e) == 'Input string cannot be empty', f"Unexpected error message: {e}"