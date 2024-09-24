from string_utils.validation import is_isbn_10

def test_is_isbn_10():
    # Test a valid ISBN-10
    assert is_isbn_10('1506715214') == True
    # Test a string shorter than 10 characters, should fail for valid ISBN-10
    assert is_isbn_10('150671521') == False
    # Test a string exactly 10 characters long with hyphens, should succeed
    assert is_isbn_10('150-6715214') == True
    # Test a string longer than 10 characters, should fail
    assert is_isbn_10('15067152145') == False