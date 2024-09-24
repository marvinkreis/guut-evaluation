from string_utils.manipulation import roman_encode, roman_decode

def test_roman_encode_decode():
    # Test case for encoding
    input_number = 4  # Roman numeral for 4 is 'IV'
    expected_roman_output = 'IV'
    
    # Encoding
    roman_output = roman_encode(input_number)
    assert roman_output == expected_roman_output, f"Expected {expected_roman_output}, but got {roman_output}"

    # Test case for decoding the roman numeral back to integer
    decoded_output = roman_decode(roman_output)
    assert decoded_output == input_number, f"Expected {input_number}, but got {decoded_output}"

    # Additional test case to expose the mutant logic
    input_number_two = 6  # Roman numeral for 6 is 'VI'
    expected_roman_output_two = 'VI'
    
    # Encoding
    roman_output_two = roman_encode(input_number_two)
    assert roman_output_two == expected_roman_output_two, f"Expected {expected_roman_output_two}, but got {roman_output_two}"

    # Test case for decoding the second roman numeral wrongfully due to mutant logic
    incorrect_decoded_output = roman_decode(roman_output_two)
    assert incorrect_decoded_output == input_number_two, f"Expected {input_number_two}, but got {incorrect_decoded_output}"