from string_utils.generation import roman_range

def test_roman_range_invalid_stop_mutant_killing():
    """
    Test the roman_range function with an invalid upper limit of 4000.
    The baseline should raise ValueError with a specific message,
    whereas the mutant should raise a ValueError with a different message.
    """
    try:
        output = list(roman_range(4000))
    except ValueError as ve:
        # Check for the specific message for the baseline
        assert str(ve) == '"stop" must be an integer in the range 1-3999', f"Unexpected error message: {str(ve)}"
    except OverflowError as oe:
        assert False, f"Raised OverflowError (not expected): {oe}"