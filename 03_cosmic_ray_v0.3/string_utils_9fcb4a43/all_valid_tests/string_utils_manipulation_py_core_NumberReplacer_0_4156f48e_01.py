from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant changes the unit mapping and should fail for input 1."""
    # Testing input of 1 which should return 'I' in Roman numeral
    output = roman_encode(1)
    assert output == 'I', "Expected 'I' for roman_encode(1) but got a different result"