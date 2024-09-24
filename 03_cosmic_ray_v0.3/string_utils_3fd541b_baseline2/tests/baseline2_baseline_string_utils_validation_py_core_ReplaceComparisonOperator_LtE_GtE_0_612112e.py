from string_utils.validation import is_ip_v4

def test_is_ip_v4():
    # Valid IP address, should return True
    assert is_ip_v4('192.168.1.1') == True
    # Valid IP address, should return True
    assert is_ip_v4('255.255.255.255') == True
    # Invalid IP address, should return False (999 is out of range)
    assert is_ip_v4('255.200.100.999') == False
    # Invalid IP address, should return False (256 is out of range)
    assert is_ip_v4('256.100.100.100') == False
    # Invalid input (not an IP), should return False
    assert is_ip_v4('nope') == False
    # Edge case of single numbers, should return False
    assert is_ip_v4('1.2.3') == False
    # Edge case of empty string, should return False
    assert is_ip_v4('') == False