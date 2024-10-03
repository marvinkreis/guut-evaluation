from string_utils.validation import is_ip_v4

def test_is_ip_v4_mutant_killing():
    """
    Test the is_ip_v4 function using the IP address 255.255.255.255.
    The baseline will return True for this valid IP, while the mutant will return False.
    """
    output = is_ip_v4('255.255.255.255')
    assert output == True, f"Expected True, got {output}"