from string_utils.validation import is_ip_v4

def test__is_ip_v4():
    """
    Test whether the function correctly identifies a valid IP v4 address.
    This test uses the IP address '0.0.0.0'. In the original code, the condition checks if each segment 
    is within the range [0, 255], so '0.0.0.0' is valid. The mutant changes the condition to (0 < int(token) <= 255),
    which makes '0.0.0.0' invalid because '0' is not strictly greater than '0'.
    """
    output = is_ip_v4('0.0.0.0')
    assert output == True  # Expected to be True for the baseline