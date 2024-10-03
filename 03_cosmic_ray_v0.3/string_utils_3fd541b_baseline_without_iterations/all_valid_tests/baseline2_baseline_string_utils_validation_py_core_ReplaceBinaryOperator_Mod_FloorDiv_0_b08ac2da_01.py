from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    # This ISBN-10 should be valid
    valid_isbn_10 = '0306406152'
    assert is_isbn_10(valid_isbn_10) == True
    
    # This ISBN-10 should be invalid
    invalid_isbn_10 = '1234567890'
    assert is_isbn_10(invalid_isbn_10) == False

    # Additional tests to ensure correctness
    another_valid_isbn_10 = '1506715214'  # This is a valid example
    assert is_isbn_10(another_valid_isbn_10) == True
    
    another_invalid_isbn_10 = '150-6715214'  # The same number with hyphen, should also be valid
    assert is_isbn_10(another_invalid_isbn_10) == True
    
    # Test an edge case with invalid number
    edge_case_invalid_isbn_10 = '123456789X'  # Invalid ISBN-10 (X is not handled)
    assert is_isbn_10(edge_case_invalid_isbn_10) == False