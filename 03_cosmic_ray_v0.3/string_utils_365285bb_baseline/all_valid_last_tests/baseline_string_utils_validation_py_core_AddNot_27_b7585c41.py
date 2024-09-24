from string_utils.validation import contains_html, InvalidInputError

def test_contains_html():
    # This test expects an InvalidInputError to be raised for non-string inputs
    try:
        contains_html(123)  # Invalid input, should raise an error
        assert False, "Expected InvalidInputError for non-string input"
    except InvalidInputError:
        pass  # This is expected

    # Valid input test
    assert contains_html('my string is <strong>bold</strong>') is True  # Contains HTML
    assert contains_html('my string is not bold') is False  # Does not contain HTML

    # Edge case test
    assert contains_html('') is False  # Empty string should return False
    assert contains_html('No tags here!') is False  # Plain text without HTML tags