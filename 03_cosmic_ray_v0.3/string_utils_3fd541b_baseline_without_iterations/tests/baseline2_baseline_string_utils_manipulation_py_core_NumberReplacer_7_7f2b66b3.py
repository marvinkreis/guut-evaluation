from string_utils.manipulation import roman_encode

def test_roman_encode():
    # The input 40 should return 'XL' in Roman numerals
    result = roman_encode(40)
    assert result == 'XL', f"Expected 'XL' but got '{result}'"
    
    # Additional tests for coverage
    result = roman_encode(10)
    assert result == 'X', f"Expected 'X' but got '{result}'"
    
    result = roman_encode(20)
    assert result == 'XX', f"Expected 'XX' but got '{result}'"
    
    result = roman_encode(30)
    assert result == 'XXX', f"Expected 'XXX' but got '{result}'"
    
    result = roman_encode(50)
    assert result == 'L', f"Expected 'L' but got '{result}'"