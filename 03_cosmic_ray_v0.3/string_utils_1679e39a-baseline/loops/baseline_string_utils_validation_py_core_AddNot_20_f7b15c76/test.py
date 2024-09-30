from string_utils.validation import is_ip_v4

def test__is_ip_v4():
    """
    Test whether an IP v4 string is validated correctly when the string is '255.200.100.75'.
    The expected outcome is True, since it's a valid IP address. With the mutant, 
    the check for ensuring each token is in the valid range (0 to 255) will fail, leading to an 
    incorrect result (False).
    """
    output = is_ip_v4('255.200.100.75')
    assert output == True