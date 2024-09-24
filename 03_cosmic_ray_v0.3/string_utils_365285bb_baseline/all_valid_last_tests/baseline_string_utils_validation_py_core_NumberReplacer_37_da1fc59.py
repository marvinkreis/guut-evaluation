from string_utils.validation import is_ip_v4

def test_is_ip_v4():
    """
    Tests the IP address validation function with valid and invalid cases including checks specifically for negatives.
    """

    # Valid IP addresses should return True
    assert is_ip_v4('192.168.0.1') == True
    assert is_ip_v4('255.255.255.255') == True
    assert is_ip_v4('0.0.0.0') == True
    assert is_ip_v4('127.0.0.1') == True  # Loopback address

    # Invalid cases that should return False
    assert is_ip_v4('192.168.0.256') == False  # Out of range
    assert is_ip_v4('192.168.0.0') == True      # Valid case to differentiate

    # Directly checking negative scenarios (focusing on mutants):
    assert is_ip_v4('192.168.0.-1') == False    # Should fail for negative
    assert is_ip_v4('-1.2.3.4') == False        # Should fail (mutant should incorrectly accept)
    assert is_ip_v4('192.-168.0.1') == False    # Another variant
    assert is_ip_v4('192.168.-1.1') == False     # Negative in the third segment
    assert is_ip_v4('-1.-1.-1.-1') == False      # All negative

    # Additional invalid cases
    assert is_ip_v4('192.168.0') == False        # Incomplete address
    assert is_ip_v4('not.an.ip') == False        # Invalid format
    assert is_ip_v4('192.168.0.1.1') == False    # More than 4 segments
    assert is_ip_v4('256.256.256.256') == False  # All components invalid
    assert is_ip_v4('') == False                  # Empty string should return False

    print("All tests passed")


# Execute the test function
test_is_ip_v4()