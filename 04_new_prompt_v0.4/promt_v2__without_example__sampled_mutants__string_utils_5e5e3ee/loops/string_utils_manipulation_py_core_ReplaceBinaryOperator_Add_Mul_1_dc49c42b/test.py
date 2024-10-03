from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the roman_encode function for values 6, 7, and 8. 
    The expected outputs are:
    - 6 should return 'VI'
    - 7 should return 'VII'
    - 8 should return 'VIII'
    
    The mutant incorrectly processes the characters for numbers 6, 7, and 8 resulting in a TypeError.
    """
    output_6 = roman_encode(6)
    output_7 = roman_encode(7)
    output_8 = roman_encode(8)
    
    assert output_6 == 'VI'
    assert output_7 == 'VII'
    assert output_8 == 'VIII'