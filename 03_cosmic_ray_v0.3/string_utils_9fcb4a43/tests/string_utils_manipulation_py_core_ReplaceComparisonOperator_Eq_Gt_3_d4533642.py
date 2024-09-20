from string_utils.manipulation import compress

def test__compress_empty_string():
    """The mutant changes error handling in `compress` to allow empty strings, while the correct code raises a ValueError."""
    try:
        compress("")  # This should raise ValueError
    except ValueError as e:
        assert str(e) == "Input string cannot be empty", "The error message for the empty string is incorrect."
    else:
        raise AssertionError("compress did not raise a ValueError for an empty string")