from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    """Changing '==' to '>=' in the is_isbn_10 method would validate invalid ISBN numbers."""
    invalid_isbn = '1234567890'  # Invalid ISBN-10
    output = is_isbn_10(invalid_isbn)
    assert not output, "is_isbn_10 must not validate an invalid ISBN-10"