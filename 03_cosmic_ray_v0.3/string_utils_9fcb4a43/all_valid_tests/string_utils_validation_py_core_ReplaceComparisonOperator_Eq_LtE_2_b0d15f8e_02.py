from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    """
    The mutant changes the condition for identifying valid ISBN-13 numbers.
    It returns 'True' for some invalid ISBN-13 inputs when it should return 'False'.
    """
    valid_isbn = '978-3-16-148410-0'   # Valid ISBN-13

    # Previously problematic ISBNs focused; now includes checksum and clear invalid formats
    invalid_isbns = [
        '978-0-306-40615-8',  # Invalid due to checksum
        '978-0-306-40615-9',  # Invalid due to checksum but valid format
        '123456789012',       # Invalid due to length
        '978-1-234-56789-0',  # Invalid due to checksum
        '978-3-16-148410-2',  # Invalid (wrong checksum for known valid structure)
        '978-3-16-148410-X',  # Invalid character
        '978-3-16-148410-1f',  # Invalid character with noise
        '978-3-16-148410-0-a', # Invalid due to trailing invalid character
    ]

    # Iterate through all invalid ISBN numbers to confirm failure on all incorrect values
    for isbn in invalid_isbns:
        assert not is_isbn_13(isbn), f"Invalid ISBN-13 '{isbn}' should not be accepted."

    # Confirms valid ISBN acceptance
    assert is_isbn_13(valid_isbn), "Valid ISBN-13 should be accepted."

# Execute the test
test__is_isbn_13()