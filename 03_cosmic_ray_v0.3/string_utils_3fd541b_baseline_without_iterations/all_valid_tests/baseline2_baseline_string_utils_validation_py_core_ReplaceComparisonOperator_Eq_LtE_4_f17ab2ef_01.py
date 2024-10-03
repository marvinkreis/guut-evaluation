from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    # Test case with a valid ISBN-10
    valid_isbn_10 = '1506715214'
    assert is_isbn_10(valid_isbn_10) == True, f"Expected True for valid ISBN-10 '{valid_isbn_10}'"
    
    # Test case with an invalid ISBN-10
    invalid_isbn_10 = '1506715215'  # Note: this should be invalid
    assert is_isbn_10(invalid_isbn_10) == False, f"Expected False for invalid ISBN-10 '{invalid_isbn_10}'"