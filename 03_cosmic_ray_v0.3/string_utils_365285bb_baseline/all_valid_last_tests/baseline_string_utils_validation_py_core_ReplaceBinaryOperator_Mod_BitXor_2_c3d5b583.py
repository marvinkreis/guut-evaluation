from string_utils.validation import is_isbn_10

def test_is_isbn_10():
    # Test cases for valid ISBN-10
    valid_isbn_10 = '1506715214'
    valid_isbn_10_with_hyphen = '150-6715214'
    
    # Valid ISBN-10 tests
    assert is_isbn_10(valid_isbn_10) == True, "Expected True for valid ISBN-10"
    assert is_isbn_10(valid_isbn_10_with_hyphen) == True, "Expected True for valid ISBN-10 with hyphen"
    
    # Invalid ISBN-10 test
    invalid_isbn_10 = '1234567890'  # This should not be a valid ISBN-10 according to checksum
    assert is_isbn_10(invalid_isbn_10) == False, "Expected False for invalid ISBN-10"

    # Additional edge cases
    assert is_isbn_10('') == False, "Expected False for empty string"
    assert is_isbn_10('1506715214X') == False, "Expected False for invalid ISBN-10 with extra character"

# Note: Run this test function to validate against both the original and mutant code.