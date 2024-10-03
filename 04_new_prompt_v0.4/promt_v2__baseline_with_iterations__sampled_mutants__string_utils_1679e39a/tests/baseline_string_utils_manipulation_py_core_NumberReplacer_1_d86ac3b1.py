from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the roman_encode function with input 1. The expected output is 'I'. 
    The mutant changes the mapping to {0: 'I', 5: 'V'}, causing it to incorrectly 
    handle the encoding for the number 1, as it will not find a mapping for 1 and return an empty string or error.
    """
    output = roman_encode(1)
    assert output == 'I'