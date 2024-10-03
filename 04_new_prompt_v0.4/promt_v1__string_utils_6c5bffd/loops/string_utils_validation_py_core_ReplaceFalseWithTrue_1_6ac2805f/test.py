from string_utils.validation import is_isbn_10

def test__is_isbn_10_invalid_character():
    """
    Tests that the function correctly identifies when a non-digit character is included in an ISBN-10 number.
    The input '123456789X' has a non-digit character and should return false for a valid ISBN-10.
    """
    output = is_isbn_10('123456789X')
    assert output is False, f"Expected False, but got {output}"