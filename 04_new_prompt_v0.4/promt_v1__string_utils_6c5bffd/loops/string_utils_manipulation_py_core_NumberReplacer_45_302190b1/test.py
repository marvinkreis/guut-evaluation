from string_utils.manipulation import roman_encode

def test_roman_encode_zero():
    """
    Test the behavior of roman_encode when given the input 0.
    The original code should raise a ValueError, while the mutant should not.
    This test is designed to expose the mutant's failure to perform proper input validation.
    """
    try:
        # This should raise a ValueError in the Baseline
        roman_encode(0)
        # If we reach this point, the mutant didn't raise the error as expected
        assert False, "Expected ValueError was not raised."
    except ValueError as e:
        # Confirming that the error is indeed raised
        assert str(e) == 'Input must be >= 1 and <= 3999', f"Unexpected ValueError message: {str(e)}"