from string_utils.validation import is_url

def test_is_url_mutant_killing():
    """
    Test the is_url function with a valid URL. The mutant will incorrectly return False,
    while the baseline will return True. The input 'http://www.example.com' is a 
    valid URL and should trigger the expected behavior.
    """
    output = is_url('http://www.example.com')
    assert output == True, f"Expected True, got {output}"