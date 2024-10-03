from string_utils.validation import is_ip_v4

def test__is_ip_v4():
    """
    Test that the valid IP address '255.0.0.255' is correctly recognized 
    by the baseline but not by the mutant. This demonstrates that the mutant 
    has altered behavior in IP address validation.
    """
    assert is_ip_v4('255.0.0.255') == True    # should return True for both
    assert is_ip_v4('192.168.1.1') == True    # should return True for both
    assert is_ip_v4('256.100.50.25') == False  # should return False for both
    assert is_ip_v4('255.200.100.300') == False  # should return False for both