from string_utils.validation import is_ip

def test_is_ip():
    # Test case where the IP address is valid for both v4 and v6
    valid_ip_v6 = '2001:db8:85a3:0000:0000:8a2e:370:7334'  # Valid IPv6
    valid_ip_v4 = '255.200.100.75'                         # Valid IPv4
    
    # Test valid IPv4; should return True with original code but False with mutant
    assert is_ip(valid_ip_v4) == True, "Should return True for valid IPv4"
    
    # Test valid IPv6; should return True with original code but False with mutant
    assert is_ip(valid_ip_v6) == True, "Should return True for valid IPv6"
    
    # Test an invalid IP not conforming to either format
    invalid_ip = '1.2.3'
    assert is_ip(invalid_ip) == False, "Should return False for invalid IP"

    print("All tests passed.")