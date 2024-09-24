from string_utils.validation import is_isbn_10

def test_is_isbn_10():
    # Valid ISBN-10; expected to return True
    assert is_isbn_10('1506715214') == True  # Valid ISBN-10
    
    # Valid ISBN-10 with hyphens; expected to return True
    assert is_isbn_10('150-6715214') == True  # Valid ISBN-10 after normalization
    
    # Invalid ISBN-10; 10 characters but not valid (should return False)
    assert is_isbn_10('1234567890') == False  # Invalid ISBN-10, exactly 10 characters
    
    # Invalid ISBN-10; exactly 10 characters but with an invalid character (should return False)
    assert is_isbn_10('123456789X') == False  # Invalid ISBN-10, exactly 10 characters
    
    # Invalid inputs with length less than 10; should return False
    assert is_isbn_10('123456789') == False  # Invalid length of 9
    
    # Invalid ISBN-13; should return False
    assert is_isbn_10('9780134190440') == False  # This is a valid ISBN-13
    
    # Invalid ISBN-10; 11 characters long (should return False)
    assert is_isbn_10('12345678901') == False  # Length is more than 10
    
    # Invalid input; 12 characters long (should return False)
    assert is_isbn_10('123456789012') == False  # Length is greater than 10
    
    # Invalid input; 13 characters long (should return False)
    assert is_isbn_10('1234567890123') == False  # Length is greater than 10
    
    # Invalid input; special character at length 10 (should return False)
    assert is_isbn_10('123456789!') == False  # Invalid ISBN-10 with special character

    # Another case for invalid input of exactly 10 characters (pure digits)
    assert is_isbn_10('123456789A') == False  # Invalid ISBN-10; must be digits and valid format
