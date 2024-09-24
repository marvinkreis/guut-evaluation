from string_utils.validation import is_url

def test__is_url_with_allowed_schemes():
    # Test case with a URL that does not start with an allowed scheme
    input_url = 'ftp://example.com'
    allowed_schemes = ['http', 'https']
    
    # The correct implementation should return False since the URL doesn't start with an allowed scheme
    expected_result = False
    actual_result = is_url(input_url, allowed_schemes)
    
    assert actual_result == expected_result, f"Expected {expected_result} but got {actual_result} with input '{input_url}'"