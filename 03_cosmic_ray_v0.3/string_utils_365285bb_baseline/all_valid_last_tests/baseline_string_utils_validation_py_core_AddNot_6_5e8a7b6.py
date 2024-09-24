from string_utils.validation import is_url

def test_is_url():
    # Test a valid URL without leading or trailing spaces
    assert is_url('http://www.example.com') == True  # Expected to pass with the correct code

    # Test an empty string (should return False)
    assert is_url('') == False  # Expected to pass with the correct code, fail with the mutant

    # Test an invalid format (not a URL)
    assert is_url('invalid_url') == False  # Expected to pass with the correct code, fail with the mutant

    # Test a valid URL with trailing spaces (should consider it invalid in correct code)
    assert is_url('   http://www.example.com   ') == False  # Expected to fail with the mutant

    # Test a valid formatted URL
    assert is_url('http://www.example.com/test') == True  # Expected to pass with the correct code

    # Test a URL with query parameters
    assert is_url('http://example.com/test?var=value') == True  # Expected to pass with the correct code

    # Test valid URL with uppercase scheme
    assert is_url('HTTPS://EXAMPLE.COM') == True  # Expected to pass with the correct code

    # Test a clearly invalid URL format
    assert is_url('ftp://example.com') == True  # Expected to pass only if allowed in the function logic