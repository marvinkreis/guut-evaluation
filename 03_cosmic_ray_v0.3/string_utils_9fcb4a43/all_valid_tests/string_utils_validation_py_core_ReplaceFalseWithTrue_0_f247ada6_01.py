from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    """The mutant incorrectly validates an ISBN-13; it should return False for invalid ISBNs."""
    valid_isbn = '9780312498580'  # Valid ISBN-13
    invalid_isbn = '9780312498581'  # Invalid ISBN-13
    assert is_isbn_13(valid_isbn) == True, "Valid ISBN-13 should return True."
    assert is_isbn_13(invalid_isbn) == False, "Invalid ISBN-13 should return False."