from string_utils.validation import is_url

def test__is_url_mutant_kill():
    """
    Test the is_url function to ensure that a URL with an unallowed scheme fails validation.
    The input 'ftp://example.com' should return False for the baseline and True for the mutant,
    indicating the mutant's faulty logic.
    """
    input_string = 'ftp://example.com'
    allowed_schemes = ['http', 'https']
    
    output = is_url(input_string, allowed_schemes)
    assert output is False  # Expecting False from baseline