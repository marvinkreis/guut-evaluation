from string_utils.validation import is_url

def test__is_url():
    # This test is for checking a valid URL
    valid_url = "http://www.example.com"
    
    # With the original function, we expect this to return True
    assert is_url(valid_url) == True, "The valid URL should return True"
    
    # To trigger the mutant behavior, we test with an empty string.
    empty_url = ""
    
    # The original function should return False for this, as it is not a valid URL.
    assert is_url(empty_url) == False, "An empty string should return False"