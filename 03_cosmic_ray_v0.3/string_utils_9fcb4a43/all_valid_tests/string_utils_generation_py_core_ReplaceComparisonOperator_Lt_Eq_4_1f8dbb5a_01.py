from string_utils.generation import roman_range

def test__roman_range():
    """The mutant changed the validation logic, allowing invalid 'stop' values."""
    # This should raise a ValueError for the correct implementation.
    try:
        result = list(roman_range(start=3999, stop=4000, step=-1))  # Out of bounds
        assert False, "Expected ValueError for out-of-bounds stop value."
    except ValueError as e:
        assert str(e) == '"stop" must be an integer in the range 1-3999', f"Unexpected exception message: {str(e)}"