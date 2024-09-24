from string_utils.manipulation import roman_encode

def test__roman_encode_49():
    """The mutant's incorrect encoding logic should fail for input 49, which should give 'XLIX'."""
    output = roman_encode(49)
    assert output == 'XLIX', "Expected output for roman_encode(49) to be 'XLIX', but got a different result."