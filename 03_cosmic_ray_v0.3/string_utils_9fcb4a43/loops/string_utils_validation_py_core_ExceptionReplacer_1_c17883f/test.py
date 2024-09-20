from string_utils.validation import is_isbn_10

def test__is_isbn_10_mixed_invalid_input():
    """Test with a string that looks valid but contains invalid characters."""
    mixed_invalid_isbn = '123456789A'  # A string that's too long for an ISBN-10

    # Call the function and assert the expected output
    output = is_isbn_10(mixed_invalid_isbn)  # Should return False for both implementations
    assert output is False, "is_isbn_10 should return False for mixed format ISBN inputs."

# Additional testing
def test__isbn_10_invalid_zero_length():
    """Test with an empty input to ensure it fails gracefully."""
    try:
        output = is_isbn_10("")  # An empty string definitely shouldn't validate
        assert output is False, "is_isbn_10 should return False for empty input."
    except InvalidInputError:
        assert False, "Function should not raise an InvalidInputError for empty input."