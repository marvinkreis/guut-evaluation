from string_utils.manipulation import roman_encode, roman_decode

def test__roman_encoding_decoding():
    """Testing encoding and decoding of Roman numbers."""
    test_cases = {
        4: 'IV',
        9: 'IX',
        20: 'XX',
        30: 'XXX',
        37: 'XXXVII',
        58: 'LVIII',
        99: 'XCIX',
        100: 'C',
        300: 'CCC',
        500: 'D',
        1000: 'M'
    }
    
    for number, expected_roman in test_cases.items():
        encoded = roman_encode(number)
        decoded = roman_decode(encoded)
        assert encoded == expected_roman, f"Encoding of {number} failed"
        assert decoded == number, f"Decoding of {expected_roman} failed"
