from string_utils.generation import roman_range

def test_roman_range_overflow():
    try:
        # This should raise an OverflowError due to the step being greater than the stop
        result = list(roman_range(5, start=8, step=2))
        # If it does not raise, the test fails
        assert False, "Expected OverflowError due to invalid start/stop/step configuration."
    except OverflowError:
        # The expected behavior, which means the test passes
        pass