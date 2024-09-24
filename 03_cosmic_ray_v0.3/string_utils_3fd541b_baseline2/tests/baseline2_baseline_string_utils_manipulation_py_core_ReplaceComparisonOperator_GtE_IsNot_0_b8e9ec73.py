from string_utils.manipulation import roman_encode, roman_decode

def test_roman_encode_decode():
    # Test case for normal encoding and decoding
    original_number = 4
    encoded = roman_encode(original_number)
    decoded = roman_decode(encoded)
    
    # Ensure that encoding and then decoding returns the original number
    assert decoded == original_number, f"Expected {original_number}, but got {decoded}"
    
    # Test case for a number that should pass both start and end boundary checks
    original_number = 9
    encoded = roman_encode(original_number)
    decoded = roman_decode(encoded)
    
    # Again, final result should return to original input
    assert decoded == original_number, f"Expected {original_number}, but got {decoded}"
    
    # Test case for the maximum boundary
    original_number = 3999
    encoded = roman_encode(original_number)
    decoded = roman_decode(encoded)
    
    # Final assertion
    assert decoded == original_number, f"Expected {original_number}, but got {decoded}"

    # Test case for a number that has multi-digit roman representation
    original_number = 58
    encoded = roman_encode(original_number)
    decoded = roman_decode(encoded)

    # Final assertion
    assert decoded == original_number, f"Expected {original_number}, but got {decoded}"