from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    """
    Test for the is_isbn_13 function using an invalid ISBN-13 to ensure the mutant is detected.
    The baseline should correctly identify that '978-03124985' (12 characters) is not valid, while the mutant 
    will incorrectly validate it due to the changed condition which allows for lengths >= 13.
    """
    # This ISBN is valid
    valid_isbn = '978-0312498580'  # This is a valid ISBN-13
    assert is_isbn_13(valid_isbn) == True  # Expected output: True (correct)

    # This ISBN is invalid (12 characters long)
    invalid_isbn = '978-03124985'  # Invalid because it has only 12 characters
    assert is_isbn_13(invalid_isbn) == False  # Expected output: False (correct)

    # This also checks for an overly long input, which should still be valid
    long_invalid_isbn = '978-03124985800'  # Invalid because it is too long
    assert is_isbn_13(long_invalid_isbn) == False  # Should be invalid with correct code and mutant