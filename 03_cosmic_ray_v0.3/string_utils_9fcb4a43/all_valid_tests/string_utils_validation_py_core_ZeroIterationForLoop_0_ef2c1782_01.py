from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    """Mutant would incorrectly validate an invalid ISBN-13 number."""
    invalid_isbn = '9780312498581'  # This should not be a valid ISBN-13
    assert not is_isbn_13(invalid_isbn), "is_isbn_13 must return False for invalid ISBN-13"