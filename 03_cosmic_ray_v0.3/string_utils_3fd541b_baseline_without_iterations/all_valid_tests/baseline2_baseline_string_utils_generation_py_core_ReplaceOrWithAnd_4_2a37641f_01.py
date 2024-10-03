from string_utils.generation import roman_range

def test_roman_range_overflow_error():
    """Test to check that roman_range raises an OverflowError for invalid range configuration."""
    try:
        # This should raise an OverflowError because we cannot step backwards to a number greater than the start.
        for _ in roman_range(stop=5, start=1, step=-1):
            pass
    except OverflowError:
        # The expected behavior is an OverflowError, so we pass the test.
        return
    
    # If we reach this point, no error was raised, which means the test has failed.
    assert False, "Expected OverflowError was not raised."