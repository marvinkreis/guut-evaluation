from string_utils.manipulation import roman_encode

def test__roman_encode():
    # Test case for the original roman_encode functionality
    assert roman_encode(5) == 'V', "Expected output for roman_encode(5) is 'V'"
    assert roman_encode(4) == 'IV', "Expected output for roman_encode(4) is 'IV'"
    assert roman_encode(6) == 'VI', "Expected output for roman_encode(6) is 'VI'"
    assert roman_encode(7) == 'VII', "Expected output for roman_encode(7) is 'VII'"