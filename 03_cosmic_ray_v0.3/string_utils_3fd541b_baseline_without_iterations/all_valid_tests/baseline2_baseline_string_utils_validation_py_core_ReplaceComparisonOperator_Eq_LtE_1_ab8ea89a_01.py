from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    # A known valid ISBN-13.
    isbn_valid = '9780312498580'  # This should return True
    assert is_isbn_13(isbn_valid) == True, "The ISBN-13 validation for a valid number failed."

    # A known invalid ISBN-13 to ensure correct failure.
    isbn_invalid = '9780312498581'  # This should return False
    assert is_isbn_13(isbn_invalid) == False, "The ISBN-13 validation for an invalid number failed."