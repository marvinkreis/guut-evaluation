from string_utils.generation import roman_range

def test__roman_range_start_less_than_stop():
    """
    Test that the roman_range function works correctly when start is less than stop.
    The expected output should be ['IV', 'V'] for the baseline, but the mutant should raise an OverflowError.
    """
    # This should work on the baseline and fail on the mutant
    output = list(roman_range(stop=5, start=4, step=1))
    assert output == ['IV', 'V']