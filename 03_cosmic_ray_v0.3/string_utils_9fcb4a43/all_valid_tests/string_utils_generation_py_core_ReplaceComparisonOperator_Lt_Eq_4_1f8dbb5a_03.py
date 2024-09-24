from string_utils.generation import roman_range

def test__roman_range_invalid_boundaries():
    """Test for out-of-range values for stop that should raise a ValueError,
    allowing us to see if the mutant's logic fails under these conditions."""

    # This test should raise a ValueError for an invalid stop value
    try:
        result = list(roman_range(start=1, stop=4000, step=1))  # Out of bounds
        assert False, "Expected ValueError for out-of-bounds stop value."
    except ValueError as e:
        assert str(e) == '"stop" must be an integer in the range 1-3999', f"Unexpected exception message: {str(e)}"

def _test__roman_range_invalid_step_zero():
    """Test a scenario where the step is zero with valid bounds."""
    try:
        result = list(roman_range(start=1, stop=5, step=0))  # Invalid step
        assert False, "Expected ValueError for zero step value."
    except ValueError as e:
        assert str(e) == 'step must be >= 1 or <= -1', f"Unexpected exception message: {str(e)}"

def _test__roman_range_non_integer_step():
    """Test a scenario where a non-integer step value is provided."""
    try:
        result = list(roman_range(start=1, stop=5, step=1.5))  # Invalid non-integer step
        assert False, "Expected ValueError for non-integer step value."
    except ValueError as e:
        assert str(e) == 'step must be an integer', f"Unexpected exception message: {str(e)}"
