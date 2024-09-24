from string_utils.validation import is_isbn

def test__is_isbn():
    # Test with a valid ISBN 10
    valid_isbn_10 = '1506715214'  # Valid ISBN-10
    assert is_isbn(valid_isbn_10) == True  # Should return True in correct code
    
    # Test with a valid ISBN 13
    valid_isbn_13 = '9780312498580'  # Valid ISBN-13
    assert is_isbn(valid_isbn_13) == True  # Should return True in correct code
    
    # Test with a string that is neither
    invalid_isbn = '1234567890X'  # Invalid ISBN
    assert is_isbn(invalid_isbn) == False  # Should return False in correct code

# The test will pass with the original code but fail with the mutant.