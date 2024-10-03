from string_utils.manipulation import compress

def test__compress_empty_string():
    """
    Test whether the compress function raises a ValueError for an empty input string.
    The baseline should raise a ValueError while the mutant should not raise any error.
    """
    try:
        compress('')
        assert False, "ValueError not raised for empty input"
    except ValueError:
        pass  # Expected behavior in baseline