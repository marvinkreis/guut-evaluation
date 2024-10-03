from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Tests the roman_encode function for the input 1. The baseline should return 'I',
    while the mutant will throw a KeyError indicating failure due to a mutant change in mappings.
    """
    output = roman_encode(1)
    assert output == 'I'