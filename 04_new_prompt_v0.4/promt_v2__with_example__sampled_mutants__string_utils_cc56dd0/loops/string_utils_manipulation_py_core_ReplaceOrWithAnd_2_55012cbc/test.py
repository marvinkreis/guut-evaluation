from string_utils.manipulation import compress

def test_compress_invalid_compression_level():
    """
    Test the compress function with an invalid compression level (string instead of int).
    The baseline should raise a ValueError while the mutant should raise a TypeError.
    This test verifies that the mutant does not handle input validation correctly.
    """
    try:
        output = compress("test", compression_level='three')
    except ValueError as e:
        assert str(e) == 'Invalid compression_level: it must be an "int" between 0 and 9', f"Expected ValueError, got: {e}"
    except TypeError as e:
        assert False, f"Expected ValueError, but got a TypeError: {e}"
    else:
        assert False, "Expected an exception, but there was no exception raised."