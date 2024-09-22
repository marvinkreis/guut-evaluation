from string_utils.validation import is_isbn_13

def test_is_isbn_13():
    # Test case with a valid ISBN-13
    valid_isbn = '9780306406157'
    assert is_isbn_13(valid_isbn) == True  # This should pass with the correct code

    # Test case with an invalid ISBN-13
    invalid_isbn = '9780306406158'
    assert is_isbn_13(invalid_isbn) == False  # This should pass with the correct code

    # Test case with hyphen in valid ISBN-13
    test_isbn_with_hyphen = '978-0306406157'
    assert is_isbn_13(test_isbn_with_hyphen) == True  # This should pass with the correct code

    # Another valid ISBN-13 to ensure correctness
    another_valid_isbn = '9780321573513'
    assert is_isbn_13(another_valid_isbn) == True  # This should pass with the correct code

    # Non-ISBN case to ensure it returns false
    non_isbn_input = '1234567890123'  # Should not be a valid ISBN-13
    assert is_isbn_13(non_isbn_input) == False  # This should pass with the correct code

    print("All assertions passed!")
