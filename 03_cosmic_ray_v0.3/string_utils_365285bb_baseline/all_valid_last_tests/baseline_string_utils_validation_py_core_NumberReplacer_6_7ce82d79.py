from string_utils.validation import is_isbn_13

def test_is_isbn_13():
    # Valid ISBN-13 number that should return True
    valid_isbn = '9780312498580'  # This is a known valid ISBN-13
    assert is_isbn_13(valid_isbn) == True, "Expected True for valid ISBN-13."

    # Invalid ISBN-13 (the last digit altered)
    invalid_isbn = '9780312498581'  # This is an invalid ISBN-13
    assert is_isbn_13(invalid_isbn) == False, "Expected False for invalid ISBN-13."

    # Invalid ISBN with non-numeric character
    invalid_isbn_invalid_char = '97803124985a0'  # Contains a letter, hence invalid
    assert is_isbn_13(invalid_isbn_invalid_char) == False, "Expected False for ISBN-13 with non-numeric character."

    # Edge Case: Length checks (more than 13 digits)
    long_isbn = '9780312498580000'  # Too long to be a valid ISBN-13
    assert is_isbn_13(long_isbn) == False, "Expected False for too long ISBN-13."
    
    # Edge Case: Check for empty string
    empty_isbn = ''
    assert is_isbn_13(empty_isbn) == False, "Expected False for empty string as ISBN-13."

# Note: With the original code, the test should pass
# With the mutant, the calculation in the method is incorrect and will therefore fail for the valid number