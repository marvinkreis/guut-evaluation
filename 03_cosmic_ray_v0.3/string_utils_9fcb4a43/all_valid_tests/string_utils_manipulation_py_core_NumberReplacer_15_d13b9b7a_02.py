from string_utils.manipulation import roman_encode

def test__roman_encode_boundaries():
    """Tests for edge cases and validations on roman_encode function."""
    
    # Testing the smallest valid input.
    assert roman_encode(1) == 'I', "Expected I for 1"

    # Testing the maximum valid input.
    assert roman_encode(3999) == 'MMMCMXCIX', "Expected MMMCMXCIX for 3999"

    # Test for the invalid input case 0
    try:
        roman_encode(0)
        assert False, "roman_encode should raise an error for zero input."
    except ValueError as e:
        assert str(e) == 'Input must be >= 1 and <= 3999', "Expecting a ValueError for zero input."

    # Test for a negative number
    try:
        roman_encode(-5)
        assert False, "roman_encode should raise an error for negative input."
    except ValueError as e:
        assert str(e) == 'Input must be >= 1 and <= 3999', "Expecting a ValueError for negative input."