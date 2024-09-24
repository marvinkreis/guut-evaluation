from string_utils.manipulation import roman_encode

def test__roman_encode():
    """Test cases intended to expose differences between the correct implementation and the mutant."""
    
    # Test for 4, expected 'IV'
    output_4 = roman_encode(4)
    assert output_4 == "IV", "roman_encode should return 'IV' for input 4"

    # Test for 5, expected 'V'
    output_5 = roman_encode(5)
    assert output_5 == "V", "roman_encode should return 'V' for input 5"
    
    # Test for 11, expected 'XI'
    output_11 = roman_encode(11)
    assert output_11 == "XI", "roman_encode should return 'XI' for input 11"