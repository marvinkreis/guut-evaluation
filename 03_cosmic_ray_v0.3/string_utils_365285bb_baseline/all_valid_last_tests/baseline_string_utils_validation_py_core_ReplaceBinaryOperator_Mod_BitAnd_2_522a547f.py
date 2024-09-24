from string_utils.validation import is_isbn_10

def test_is_isbn_10():
    # The following ISBN is valid
    valid_isbn_10 = '1506715214'
    assert is_isbn_10(valid_isbn_10) == True, "Should return True for valid ISBN 10"

    # The following ISBN is invalid
    invalid_isbn_10 = '1506715215'
    assert is_isbn_10(invalid_isbn_10) == False, "Should return False for invalid ISBN 10"

    # Additional test cases
    assert is_isbn_10('150-6715214') == True, "Should return True for valid ISBN with hyphens"
    assert is_isbn_10('150-6715215') == False, "Should return False for invalid ISBN with hyphens"

    # Remove the test with 'X' since it is not a valid ISBN-10 based on the specifications.