from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    # Test case for a valid ISBN-13 number
    valid_isbn_13 = '9780312498580'
    assert is_isbn_13(valid_isbn_13) == True, "Expected True for valid ISBN-13"
    
    # Test case for an invalid ISBN-13 number
    invalid_isbn_13 = '9780312498581'
    assert is_isbn_13(invalid_isbn_13) == False, "Expected False for invalid ISBN-13"