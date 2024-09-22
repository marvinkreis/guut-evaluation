from string_utils.validation import is_ip_v4

def test_is_ip_v4():
    # This IP address is valid as it has the maximum value 255
    valid_ip = "255.255.255.255"
    assert is_ip_v4(valid_ip) == True, f"Test failed: {valid_ip} should be valid."

    # This IP address is invalid due to the mutant condition, we expect it to return False
    invalid_ip = "255.255.255.256"
    assert is_ip_v4(invalid_ip) == False, f"Test failed: {invalid_ip} should be invalid."

    # Test with another valid IP address
    valid_ip_2 = "192.168.1.1"
    assert is_ip_v4(valid_ip_2) == True, f"Test failed: {valid_ip_2} should be valid."

    # Test with another invalid IP address, but lower than the mutant limit
    invalid_ip_2 = "256.100.50.25"
    assert is_ip_v4(invalid_ip_2) == False, f"Test failed: {invalid_ip_2} should be invalid."