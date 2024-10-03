from string_utils.manipulation import compress

def test__compress_with_invalid_compression_level():
    """
    Test whether the compress function raises a ValueError when compression_level is set to -1.
    The original function should raise an error, while the mutant should not due to the change in condition.
    """
    try:
        compress("Sample string", compression_level=-1)
        # If this line is reached, the test should fail since no exception was raised
        assert False, "Expected ValueError was not raised."
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")