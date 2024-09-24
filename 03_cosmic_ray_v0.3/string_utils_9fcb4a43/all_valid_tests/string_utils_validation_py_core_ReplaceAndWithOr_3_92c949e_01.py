from string_utils.validation import is_url

def test__is_url():
    """This test checks the behavior of the is_url function against a mutant version."""
    
    # Valid URL
    valid_url = 'http://valid-url.com'  
    assert is_url(valid_url), "The valid URL should return True"

    # Unsupported scheme
    unsupported_scheme_url = 'file://local-file'  
    assert not is_url(unsupported_scheme_url), "The unsupported scheme URL should return False"

    # Invalid URL with special characters
    clearly_invalid_url_with_special_chars = 'http://#invalid'  
    assert not is_url(clearly_invalid_url_with_special_chars), "The clearly invalid URL should return False"

# Run the test
test__is_url()