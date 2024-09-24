from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant's change in logic should cause it to fail when encoding 14 to 'XIV'."""
    output = roman_encode(14)
    assert output == 'XIV', "Expected output for roman_encode(14) to be 'XIV', but got a different result."