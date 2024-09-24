from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant's flawed encoding should fail for input 39, which should encode to 'XXXIX'."""
    output = roman_encode(39)
    assert output == 'XXXIX', "Expected output for roman_encode(39) to be 'XXXIX', but got a different result."