from string_utils.validation import is_ip_v4

def test__is_ip_v4():
    """
    Test the is_ip_v4 function to ensure correct validation of IP addresses.
    The input '0.0.0.0' should be valid, but the mutant version incorrectly processes it.
    This test distinguishes between the baseline and mutant implementations.
    """
    # Test for valid IP addresses
    assert is_ip_v4('192.168.1.1') == True
    assert is_ip_v4('255.255.255.255') == True
    assert is_ip_v4('0.0.0.0') == True  # This should return True for both Baseline and Mutant
    
    # Test for invalid IPs
    assert is_ip_v4('256.100.50.25') == False
    assert is_ip_v4('192.168.1.256') == False
    assert is_ip_v4('-1.0.0.0') == False
    assert is_ip_v4('abc.def.ghi.jkl') == False

test__is_ip_v4()