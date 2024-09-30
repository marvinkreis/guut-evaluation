from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test whether the method correctly converts the integer 6 to the Roman numeral VI.
    The mutant incorrectly computes this by performing a subtraction instead of a multiplication,
    which will yield an incorrect Roman numeral for the value of 6.
    """
    output = roman_encode(6)
    assert output == 'VI'