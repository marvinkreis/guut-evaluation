from string_utils.validation import is_url

def test__is_url():
    """The mutant changes the logic in is_url, causing it to fail on valid URLs."""
    # Test valid URL
    assert is_url('http://example.com'), "Expected 'http://example.com' to be a valid URL"
    # Test invalid URL
    assert not is_url('invalid_url'), "Expected 'invalid_url' to not be a valid URL"
    # Test empty string
    assert not is_url(''), "Expected empty string to not be a valid URL"
    # Test whitespace
    assert not is_url('   '), "Expected string with only spaces to not be a valid URL"
    # Test well-formed URL
    assert is_url('https://www.example.com/path/to/resource'), "Expected valid URL to return True"