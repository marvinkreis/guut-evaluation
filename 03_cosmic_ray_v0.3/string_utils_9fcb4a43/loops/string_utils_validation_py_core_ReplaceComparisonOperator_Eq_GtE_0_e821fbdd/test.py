from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    """The mutant incorrectly validates ISBN-13 numbers. It accepts strings longer than 13 digits as valid."""
    valid_isbn = '9780312498580'  # Valid ISBN-13
    invalid_isbn = '97803124985800'  # Invalid ISBN-13 (length 14)
    
    assert is_isbn_13(valid_isbn), "Valid ISBN-13 should return True"
    assert not is_isbn_13(invalid_isbn), "Invalid ISBN-13 should return False"