from string_utils.validation import is_ip_v4

def test__is_ip_v4():
    # Valid IPv4 addresses
    assert is_ip_v4('192.168.1.1') == True
    assert is_ip_v4('255.255.255.255') == True
    assert is_ip_v4('0.0.0.0') == True
    
    # Invalid IPv4 addresses
    assert is_ip_v4('256.100.100.100') == False  # 256 is out of range
    assert is_ip_v4('192.168.-1.1') == False     # Negative number
    assert is_ip_v4('192.168.1.1.1') == False     # Too many segments
    assert is_ip_v4('192.168.1') == False         # Too few segments
    assert is_ip_v4('abc.def.ghi.jkl') == False   # Non-numeric values

    # Edge cases
    assert is_ip_v4('255.200.100.75') == True     # Valid IP
    assert is_ip_v4('255.200.100.999') == False    # Last octet out of range