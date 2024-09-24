from string_utils.manipulation import roman_encode

def test__roman_encode():
    # Test the encoding for the correct value of 5
    result = roman_encode(5)
    expected = "V"  # The expected result for 5 in Roman numerals
    assert result == expected, f"Expected {expected} but got {result}"

    # Test encoding of other values to ensure they still work correctly with the original code
    assert roman_encode(1) == "I", "Should be I"
    assert roman_encode(4) == "IV", "Should be IV"
    assert roman_encode(10) == "X", "Should be X"
    assert roman_encode(37) == "XXXVII", "Should be XXXVII"