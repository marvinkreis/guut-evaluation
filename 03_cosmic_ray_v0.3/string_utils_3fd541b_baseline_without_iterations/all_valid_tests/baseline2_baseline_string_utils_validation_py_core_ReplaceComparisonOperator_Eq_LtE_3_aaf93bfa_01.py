from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    # This test case checks the behavior of the is_isbn_10 function 
    # when given a string with a length of exactly 10 characters.

    valid_isbn_10 = '1506715214'  # A valid ISBN-10
    invalid_isbn_10 = '15067152145'  # An invalid ISBN-10 (length 11)
    invalid_isbn_too_short = '150671521'  # An invalid ISBN-10 (length 9)
    
    # Assert that the valid ISBN passes
    assert is_isbn_10(valid_isbn_10) == True
    
    # Assert that an invalid ISBN (too long) fails
    assert is_isbn_10(invalid_isbn_10) == False
    
    # Assert that an invalid ISBN (too short) fails
    assert is_isbn_10(invalid_isbn_too_short) == False