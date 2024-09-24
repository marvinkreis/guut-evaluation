from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    # This is a valid ISBN-10
    valid_isbn_10 = '1506715214'
    # Check the original validity
    assert is_isbn_10(valid_isbn_10) == True, "Expected True for valid ISBN-10"

    # Testing with a known invalid ISBN-10 to ensure the mutant behaves differently
    invalid_isbn_10 = '1506715215'  # This is an invalid ISBN
    assert is_isbn_10(invalid_isbn_10) == False, "Expected False for invalid ISBN-10"

# To execute the test, simply call the function:
# test__is_isbn_10()