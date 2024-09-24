from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Test the encoding of 1000, should return 'M'
    result = roman_encode(1000)
    expected = 'M'
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test encoding of values that yield multiple symbols
    assert roman_encode(1987) == 'MCMLXXXVII', "Encoding for 1987 should be 'MCMLXXXVII'"
    
    # This case is a critical test since the mutation would fail to encode 2000 correctly.
    assert roman_encode(2000) == 'MM', "Encoding for 2000 should be 'MM'"

    # Additional cases for validation
    assert roman_encode(1) == 'I', "Encoding for 1 should be 'I'"
    assert roman_encode(4) == 'IV', "Encoding for 4 should be 'IV'"
    assert roman_encode(5) == 'V', "Encoding for 5 should be 'V'"
    assert roman_encode(10) == 'X', "Encoding for 10 should be 'X'"
    assert roman_encode(37) == 'XXXVII', "Encoding for 37 should be 'XXXVII'"
    assert roman_encode(2021) == 'MMXXI', "Encoding for 2021 should be 'MMXXI'"
    assert roman_encode(3999) == 'MMMCMXCIX', "Encoding for 3999 should be 'MMMCMXCIX'"