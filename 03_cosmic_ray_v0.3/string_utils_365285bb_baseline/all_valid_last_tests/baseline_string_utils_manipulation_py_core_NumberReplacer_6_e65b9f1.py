from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Testing the roman encoding for the integer 60
    result = roman_encode(60)
    expected_result = 'LX'  # Correct encoding for 60 in Roman numerals
    
    assert result == expected_result, f"Expected {expected_result} but got {result}"