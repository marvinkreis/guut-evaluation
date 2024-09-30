from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test encoding of the number 4 to ensure that it produces the correct Roman numeral 'IV'.
    The mutant changes the encoding rules such that the numeral for 4 would incorrectly be 'V'. This test
    case will pass with the original implementation and fail with the mutant.
    """
    output = roman_encode(4)
    assert output == 'IV'