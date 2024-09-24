from string_utils.manipulation import roman_encode

def test__roman_encode():
    # Test valid input that the correct function can accurately convert to Roman numeral
    assert roman_encode(200) == 'CC'  # This should return 'CC' for 200
    assert roman_encode(37) == 'XXXVII'  # This should return 'XXXVII' for 37
    assert roman_encode(1) == 'I'  # This should return 'I' for 1
    assert roman_encode(3999) == 'MMMCMXCIX'  # This should return 'MMMCMXCIX' for 3999