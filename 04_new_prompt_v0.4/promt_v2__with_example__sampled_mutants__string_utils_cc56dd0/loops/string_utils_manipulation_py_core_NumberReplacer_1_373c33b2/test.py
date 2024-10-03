from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function with the input 1. The baseline should return 'I', which represents 
    the roman numeral for 1. The mutant, however, will raise a KeyError because it incorrectly maps
    'I' to the key 0 instead of 1.
    """
    output = roman_encode(1)
    assert output == 'I', f"Expected 'I', got {output}"