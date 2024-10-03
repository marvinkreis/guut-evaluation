from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the roman_encode function with the integer 40. The correct encoding for 40 is 'XL'.
    The mutant changes the mapping for 40 to be incorrectly encoded, so it will produce a different output,
    which allows us to detect the mutant.
    """
    output = roman_encode(40)
    assert output == 'XL'