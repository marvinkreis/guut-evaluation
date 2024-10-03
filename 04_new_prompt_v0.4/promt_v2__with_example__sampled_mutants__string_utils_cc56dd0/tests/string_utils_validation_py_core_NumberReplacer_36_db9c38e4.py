from string_utils.validation import is_ip_v4

def test_is_ip_v4_mutant_killing():
    """
    Test the is_ip_v4 function with '0.0.0.0', which is a valid IPv4 address.
    The mutant will incorrectly return False, while the baseline will return True.
    """
    output = is_ip_v4('0.0.0.0')
    assert output == True, f"Expected True, got {output}"