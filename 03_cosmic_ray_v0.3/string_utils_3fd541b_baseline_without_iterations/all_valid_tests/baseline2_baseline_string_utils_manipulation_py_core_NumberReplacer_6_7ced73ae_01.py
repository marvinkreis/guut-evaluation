from string_utils.manipulation import roman_encode

def test__roman_encode():
    # Testing the encoding for the integer value 40
    result = roman_encode(40)
    expected = 'XL'  # Expecting XL for the input 40
    assert result == expected, f"Expected {expected}, but got {result}"