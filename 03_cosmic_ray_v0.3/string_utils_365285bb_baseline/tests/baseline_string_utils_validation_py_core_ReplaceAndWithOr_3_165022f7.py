from string_utils.validation import is_url

def test_is_url():
    # Test cases with valid URLs and a set of allowed schemes
    assert is_url('http://www.example.com', allowed_schemes=['http', 'https']) == True  # Should pass
    assert is_url('https://www.example.com', allowed_schemes=['http', 'https']) == True  # Should pass
    assert is_url('ftp://ftp.example.com', allowed_schemes=['http', 'https']) == False  # Should fail because 'ftp' is not in allowed schemes
    assert is_url('http://www.example.com', allowed_schemes=['ftp']) == False  # Should fail because 'http' is not in allowed schemes

    # Invalid URL with allowed schemes
    assert is_url('.example.com', allowed_schemes=['http', 'https']) == False  # Should fail as it is not valid

    # Test cases without allowed schemes
    assert is_url('http://www.example.com') == True  # Should pass
    assert is_url('https://www.example.com') == True  # Should pass
    
    # Edge case with empty string
    assert is_url('', allowed_schemes=['http', 'https']) == False  # Should fail because it's not a valid URL

# The test function will pass in the correct code but fail with the mutant.