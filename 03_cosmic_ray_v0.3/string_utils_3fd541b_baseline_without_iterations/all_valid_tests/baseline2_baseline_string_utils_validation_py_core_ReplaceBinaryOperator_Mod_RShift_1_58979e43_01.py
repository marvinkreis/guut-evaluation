from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    # Valid ISBN-10 tests
    assert is_isbn_10('1506715214') == True  # Valid ISBN-10
    assert is_isbn_10('0-306-40615-2') == True  # Valid ISBN-10 with hyphens
    assert is_isbn_10('9971-5-0210-0') == True  # Valid ISBN-10 with hyphens

    # Invalid ISBN-10 tests
    assert is_isbn_10('123456789X') == False  # Invalid because of non-numeric ending
    assert is_isbn_10('1234567890') == False  # Invalid checksum
    assert is_isbn_10('150-6715215') == False  # Valid format but invalid ISBN-10
    
    # Edge cases
    assert is_isbn_10('') == False  # Empty string
    assert is_isbn_10('123') == False  # Too short
    
    print("All tests passed.")