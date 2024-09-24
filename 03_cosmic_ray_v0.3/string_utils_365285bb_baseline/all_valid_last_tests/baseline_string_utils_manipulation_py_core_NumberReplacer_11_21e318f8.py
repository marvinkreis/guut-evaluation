from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Testing the encoding of the number 400
    result = roman_encode(400)
    expected = 'CD'  # CD is the correct encoding for 400
    assert result == expected, f"Expected {expected}, but got {result}"

    # Additional tests to ensure the integrity of the encoding
    assert roman_encode(1) == 'I', "1 should be encoded as I"
    assert roman_encode(5) == 'V', "5 should be encoded as V"
    assert roman_encode(10) == 'X', "10 should be encoded as X"
    assert roman_encode(50) == 'L', "50 should be encoded as L"
    assert roman_encode(100) == 'C', "100 should be encoded as C"
    assert roman_encode(500) == 'D', "500 should be encoded as D"
    assert roman_encode(1000) == 'M', "1000 should be encoded as M"