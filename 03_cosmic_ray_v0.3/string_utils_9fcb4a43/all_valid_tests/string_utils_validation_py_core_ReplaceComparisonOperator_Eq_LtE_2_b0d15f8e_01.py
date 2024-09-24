from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    """
    The mutant changes the condition for identifying valid ISBN-13 numbers.
    It returns 'True' for some invalid ISBN-13 inputs when it should return 'False'.
    """
    # Known valid and invalid ISBN-13 numbers
    invalid_isbns = [
        '978-0-306-40615-8',  # Invalid ISBN-13 (checksum failure)
        '978-1-234-56789-0',  # Invalid ISBN-13 (checksum failure)
        '978-3-16-148410-7',  # Valid ISBN-13 but with an incorrect checksum for testing
        '1234567890123',      # Invalid format (not 13 characters)
        '9783061484100',      # Invalid ISBN-13 (should return false)
        '978-3-16-148410-0a',  # Invalid due to extra character
    ]
    valid_isbn = '978-3-16-148410-0'  # Valid ISBN-13

    for isbn in invalid_isbns:
        assert not is_isbn_13(isbn), f"Invalid ISBN-13 '{isbn}' should not be accepted."

    assert is_isbn_13(valid_isbn), "Valid ISBN-13 should be accepted."