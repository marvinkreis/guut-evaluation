from string_utils.manipulation import roman_encode

def test_roman_encode():
    # The correct encoding for 5 should be 'V'
    assert roman_encode(5) == 'V', "Test failed: Expected 'V' for input 5"
    # Additional tests to ensure correct encoding for other values
    assert roman_encode(1) == 'I', "Test failed: Expected 'I' for input 1"
    assert roman_encode(2) == 'II', "Test failed: Expected 'II' for input 2"
    assert roman_encode(3) == 'III', "Test failed: Expected 'III' for input 3"
    assert roman_encode(4) == 'IV', "Test failed: Expected 'IV' for input 4"
    assert roman_encode(6) == 'VI', "Test failed: Expected 'VI' for input 6"
    assert roman_encode(7) == 'VII', "Test failed: Expected 'VII' for input 7"
    assert roman_encode(8) == 'VIII', "Test failed: Expected 'VIII' for input 8"
    assert roman_encode(9) == 'IX', "Test failed: Expected 'IX' for input 9"
    assert roman_encode(10) == 'X', "Test failed: Expected 'X' for input 10"
    assert roman_encode(37) == 'XXXVII', "Test failed: Expected 'XXXVII' for input 37"