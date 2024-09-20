from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    """The mutant changes the required length for a valid ISBN-13, causing it to fail for valid inputs."""
    valid_isbn_13 = '9780312498580'
    assert is_isbn_13(valid_isbn_13) is True, "is_isbn_13 must validate the provided ISBN-13 correctly."