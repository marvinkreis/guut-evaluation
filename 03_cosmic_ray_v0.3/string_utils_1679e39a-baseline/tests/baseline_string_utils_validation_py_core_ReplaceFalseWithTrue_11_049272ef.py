from string_utils.validation import is_ip_v4

def test__is_ip_v4():
    """
    Test whether an input string that is clearly not an IPv4 address is correctly identified as invalid. The input 'not.an.ip' 
    should return False for the baseline implementation, which checks for valid IPv4 formats. The mutant, however, will incorrectly 
    return True due to the flawed condition in the initial validation check.
    """
    output = is_ip_v4('not.an.ip')
    assert output is False