from string_utils.validation import is_url

def test__is_url_mutant_kill():
    """
    Test to kill the mutant by checking if valid URLs with unallowed schemes
    correctly return False in the baseline but True in the mutant.
    Here, we are validating a URL with 'http' scheme against allowed schemes 
    that only include 'https', expecting False for the baseline and True for the mutant.
    """
    output = is_url('http://www.example.com', allowed_schemes=['https'])
    assert output == False