from string_utils.manipulation import roman_encode

def test__roman_encode_zero():
    """
    Test roman_encode function with input 0.
    The baseline should raise ValueError for this invalid input.
    The mutant should accept this input and return None (or an empty string).
    """
    # Attempt to encode 0
    try:
        roman_encode(0)
        # If we reach here, that means mutant accepted the input
        assert False, "Expected ValueError but got a valid output instead."
    except ValueError as e:
        assert str(e) == 'Input must be >= 1 and <= 3999', "Baseline should raise ValueError for invalid input"