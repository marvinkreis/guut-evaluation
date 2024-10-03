from string_utils.manipulation import roman_encode, roman_decode

def test__roman_encode_decode():
    # Test sequence to verify correct encoding and decoding of a roman numeral
    
    # Test valid encodings
    assert roman_encode(5) == 'V'  # Encoding for 5
    assert roman_encode(6) == 'VI'  # Encoding for 6
    assert roman_encode(7) == 'VII'  # Encoding for 7
    assert roman_encode(8) == 'VIII'  # Encoding for 8
    assert roman_encode(9) == 'IX'  # Encoding for 9
    assert roman_encode(10) == 'X'  # Encoding for 10
    assert roman_encode(37) == 'XXXVII'  # Encoding for 37

    # Test decoding and ensure it matches the original numbers
    assert roman_decode('V') == 5  # Decoding for V
    assert roman_decode('VI') == 6  # Decoding for VI
    assert roman_decode('VII') == 7  # Decoding for VII
    assert roman_decode('VIII') == 8  # Decoding for VIII
    assert roman_decode('IX') == 9  # Decoding for IX
    assert roman_decode('X') == 10  # Decoding for X
    assert roman_decode('XXXVII') == 37  # Decoding for XXXVII

    # Test edge cases
    assert roman_encode(1) == 'I'  # Edge case for 1
    assert roman_encode(3999) == 'MMMCMXCIX'  # Edge case for 3999
    assert roman_decode('I') == 1  # Decoding for I
    assert roman_decode('MMMCMXCIX') == 3999  # Decoding for MMMCMXCIX

# Running the test to ensure it catches mutant behavior
test__roman_encode_decode()