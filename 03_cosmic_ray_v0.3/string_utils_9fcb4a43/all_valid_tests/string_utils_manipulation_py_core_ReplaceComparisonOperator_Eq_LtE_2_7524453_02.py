from string_utils.manipulation import roman_encode

def test__roman_encode():
    """Testing roman encoding for values that may reveal mutant behavior."""
    
    # Test for value 4, expected 'IV'
    output_4 = roman_encode(4)
    assert output_4 == "IV", "roman_encode should return 'IV' for the input 4"
    
    # Test for value 6, expected 'VI'
    output_6 = roman_encode(6)
    assert output_6 == "VI", "roman_encode should return 'VI' for the input 6"