from string_utils.manipulation import roman_encode, roman_decode

def test__roman_encode_and_decode():
    # Test case for encoding a valid integer to Roman numeral
    original_number = 10
    encoded_number = roman_encode(original_number)
    
    # The expected output is 'X' for the number 10
    assert encoded_number == 'X', f"Expected 'X' but got {encoded_number}"

    # Test case for decoding the Roman numeral back to an integer
    decoded_number = roman_decode(encoded_number)

    # The expected output is 10
    assert decoded_number == original_number, f"Expected {original_number} but got {decoded_number}"

    # Testing another number to ensure robustness
    another_number = 39
    encoded_another = roman_encode(another_number)
    
    # The expected output is 'XXXIX' for the number 39
    assert encoded_another == 'XXXIX', f"Expected 'XXXIX' but got {encoded_another}"

    decoded_another = roman_decode(encoded_another)

    # The expected output is 39
    assert decoded_another == another_number, f"Expected {another_number} but got {decoded_another}"

# Invoke the test
test__roman_encode_and_decode()