from string_utils.manipulation import roman_encode, roman_decode

def test_roman_encode_decode():
    # Testing the encoding of numbers that are sensitive to known mappings.
    # This includes the direct numbers and their edge cases.

    test_cases = {
        1: 'I',
        2: 'II',
        3: 'III',
        4: 'IV',  # This will be sensitive to the mapping change
        5: 'V',   # This tests the mutant directly
        6: 'VI',
        9: 'IX',
        10: 'X',
        39: 'XXXIX', 
        40: 'XL',   # This should also validate the 5 mapping as it leads to 'L'
        50: 'L',
        100: 'C',
        500: 'D',
        1000: 'M',
        3999: 'MMMCMXCIX'
    }

    for num, expected in test_cases.items():
        # Test the encoding
        encoded_value = roman_encode(num)
        assert encoded_value == expected, f"Expected '{expected}' for {num}, but got '{encoded_value}'"

        # Confirm correct decoding back to the original number
        decoded_value = roman_decode(encoded_value)
        assert decoded_value == num, f"Expected {num}, but got {decoded_value}"

# This case explicitly focuses on encoding that leverages the mapping change