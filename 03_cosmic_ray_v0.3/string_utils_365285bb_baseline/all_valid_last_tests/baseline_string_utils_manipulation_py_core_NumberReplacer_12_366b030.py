from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Test case for the correct encoding of the number 2000
    result = roman_encode(2000)
    # It should return the correct Roman numeral for 2000, which is "MM"
    assert result == "MM", f"Expected 'MM', but got '{result}'"

    # Additional checks to ensure the encoding is properly handled
    # This is to ensure robustness; feels free to expand it further
    assert roman_encode(1) == "I", "1 should be encoded to 'I'"
    assert roman_encode(5) == "V", "5 should be encoded to 'V'"
    assert roman_encode(10) == "X", "10 should be encoded to 'X'"
    assert roman_encode(1999) == "MCMXCIX", "1999 should be encoded to 'MCMXCIX'"