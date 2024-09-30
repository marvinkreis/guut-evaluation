from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test if the roman_encode function correctly encodes the number 8.
    The input of 8 should return 'VIII' in standard Roman numeral notation. 
    The mutant changes the condition leading to incorrect encodings for values
    greater than 7. If the condition is altered, the output for 8 would be 'VA',
    which is incorrect.
    """
    output = roman_encode(8)
    assert output == 'VIII'