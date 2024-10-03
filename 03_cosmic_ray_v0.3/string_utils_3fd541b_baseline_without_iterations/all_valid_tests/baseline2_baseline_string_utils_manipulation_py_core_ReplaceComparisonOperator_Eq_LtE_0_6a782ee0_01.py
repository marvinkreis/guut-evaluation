from string_utils.manipulation import roman_encode

def test__roman_encode_zero():
    # Test input of 0 which should raise a ValueError in the original implementation
    try:
        result = roman_encode(0)
        assert False, f"Expected ValueError not raised, got {result}"
    except ValueError as e:
        assert str(e) == 'Input must be >= 1 and <= 3999', f"Unexpected exception message: {str(e)}"
    
    # Also testing for a negative number
    try:
        result = roman_encode(-1)
        assert False, f"Expected ValueError not raised, got {result}"
    except ValueError as e:
        assert str(e) == 'Input must be >= 1 and <= 3999', f"Unexpected exception message: {str(e)}"
