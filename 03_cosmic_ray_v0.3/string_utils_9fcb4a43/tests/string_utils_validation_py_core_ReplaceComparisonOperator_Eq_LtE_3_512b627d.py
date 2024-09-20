from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    """The mutant has a faulty check that allows invalid ISBNs of less than 10 digits to be accepted."""
    # We expect this input to return False for a valid ISBN check
    assert is_isbn_10('0') is False, "Single digit '0' should not be recognized as a valid ISBN-10"
    assert is_isbn_10('123456789') is False, "Nine digits should not be recognized as a valid ISBN-10"
    assert is_isbn_10('12345678') is False, "Eight digits should not be recognized as a valid ISBN-10"