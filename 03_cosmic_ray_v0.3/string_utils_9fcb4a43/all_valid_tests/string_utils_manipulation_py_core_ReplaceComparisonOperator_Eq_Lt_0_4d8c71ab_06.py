from string_utils.manipulation import roman_encode

def test__roman_encode_34():
    """The mutant's output should fail when encoding 34, which should yield 'XXXIV'."""
    output = roman_encode(34)
    assert output == 'XXXIV', "Expected output for roman_encode(34) to be 'XXXIV', but got a different result."