from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant's flawed encoding logic should fail when encoding 44 to 'XLIV'."""
    output = roman_encode(44)
    assert output == 'XLIV', "Expected output for roman_encode(44) to be 'XLIV', but got a different result."