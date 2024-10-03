from string_utils.generation import roman_range

def test__roman_range_for_too_high_input():
    """
    Test whether the `roman_range` function raises a ValueError when the input exceeds the valid range for Roman numerals.
    Here, we check for the input 4000, which should raise a ValueError in the baseline, and will appropriately pass through
    the mutant due to altered validation logic. The expected error message is checked to verify correctness.
    """
    try:
        for _ in roman_range(4000):
            pass
        assert False, "ValueError not raised for input 4000"
    except ValueError as e:
        assert str(e) == '"stop" must be an integer in the range 1-3999'  # This will pass for the baseline and fail for the mutant.