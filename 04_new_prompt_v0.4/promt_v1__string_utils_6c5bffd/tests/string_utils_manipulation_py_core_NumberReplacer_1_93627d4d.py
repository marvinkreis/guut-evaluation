from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the roman_encode function for an input of 1. The baseline should return 'I' for the input,
    while the mutant will fail due to a KeyError because of the change in the __mappings, which affects
    how the number 1 is represented in Roman numerals.
    """
    output = roman_encode(1)
    assert output == 'I'