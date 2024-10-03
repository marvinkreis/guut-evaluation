from string_utils.validation import is_isbn

def test__is_isbn():
    # Test cases for valid ISBN-10
    valid_isbn_10 = '1506715214'
    assert is_isbn(valid_isbn_10) == True, "Test case for valid ISBN-10 failed"

    # Test cases for valid ISBN-13
    valid_isbn_13 = '9780312498580'
    assert is_isbn(valid_isbn_13) == True, "Test case for valid ISBN-13 failed"

    # Test case for a string that is not an ISBN
    invalid_isbn = 'not_an_isbn'
    assert is_isbn(invalid_isbn) == False, "Test case for invalid ISBN failed"