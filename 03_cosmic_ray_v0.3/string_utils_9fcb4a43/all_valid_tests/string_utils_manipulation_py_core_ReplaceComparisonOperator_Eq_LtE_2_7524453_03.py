from string_utils.manipulation import roman_encode

def test__roman_encode():
    """Test cases that exploit the mutant's behavioral changes."""
    
    # Test for value 3, expected 'III'
    output_3 = roman_encode(3)
    assert output_3 == "III", "roman_encode should return 'III' for the input 3"
    
    # Test for value 1, expected 'I'
    output_1 = roman_encode(1)
    assert output_1 == "I", "roman_encode should return 'I' for the input 1"
    
    # Test for value 4, expected 'IV'
    output_4 = roman_encode(4)
    assert output_4 == "IV", "roman_encode should return 'IV' for the input 4"