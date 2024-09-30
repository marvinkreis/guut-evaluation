from string_utils.validation import is_url

def test__is_url():
    """
    Test whether the function accurately identifies valid URLs when allowed schemes are specified. 
    Input 'ftp://www.example.com' should return False when checking against allowed schemes 
    that include only 'http', since 'ftp' is not an allowed scheme. The mutant's change from 
    'and' to 'or' would incorrectly allow this URL to pass, thus leading to a failed test.
    """
    output = is_url('ftp://www.example.com', allowed_schemes=['http'])
    assert output == False