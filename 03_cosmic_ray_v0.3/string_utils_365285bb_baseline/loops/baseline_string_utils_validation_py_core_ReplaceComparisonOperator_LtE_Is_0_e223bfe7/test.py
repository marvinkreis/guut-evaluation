from string_utils.validation import is_ip_v4

def test_is_ip_v4():
    # Test case where the original code should return True for a valid IP address
    result_valid_ip = is_ip_v4('192.168.1.1')
    assert result_valid_ip == True, "Expected True for valid IP address"

    # Test case where the original code should return False for an invalid IP format
    result_invalid_ip = is_ip_v4('192.168.256.1')
    assert result_invalid_ip == False, "Expected False for invalid IP address"

    # Test case where the original code should return False for a malformed IP address with empty octet
    result_invalid_token = is_ip_v4('192.168..1')
    assert result_invalid_token == False, "Expected False for IP with empty octet"