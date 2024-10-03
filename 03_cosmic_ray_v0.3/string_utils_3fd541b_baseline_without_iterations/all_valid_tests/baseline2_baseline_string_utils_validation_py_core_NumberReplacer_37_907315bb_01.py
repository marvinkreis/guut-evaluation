from string_utils.validation import is_ip_v4

def test_is_ip_v4():
    # Valid IPv4 addresses
    assert is_ip_v4('192.168.1.1') == True
    assert is_ip_v4('255.255.255.255') == True
    
    # Invalid IPv4 addresses
    assert is_ip_v4('256.100.50.25') == False  # 256 is out of range
    assert is_ip_v4('192.168.1.300') == False  # 300 is out of range
    assert is_ip_v4('192.168.-1.1') == False  # -1 is not a valid segment
    
    # Edge cases
    assert is_ip_v4('0.0.0.0') == True  # 0 is a valid segment
    assert is_ip_v4('192.168.1.') == False  # Trailing dot is invalid
    assert is_ip_v4('192.168.1.1.1') == False  # Too many segments