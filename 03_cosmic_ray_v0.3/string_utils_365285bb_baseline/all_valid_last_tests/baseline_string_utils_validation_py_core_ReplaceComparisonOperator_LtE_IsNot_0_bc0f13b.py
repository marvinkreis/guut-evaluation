from string_utils.validation import is_ip_v4

def test__is_ip_v4():
    # Test a valid IPv4 address
    valid_ip = '192.168.1.1'
    assert is_ip_v4(valid_ip) == True, f"Expected True for valid IP {valid_ip}"

    # Test an IP address with a segment out of the valid range (256 is out of range)
    invalid_ip = '256.100.50.25'
    assert is_ip_v4(invalid_ip) == False, f"Expected False for invalid IP {invalid_ip}"

    # Special case: Test an IP address consisting entirely of zeros
    zero_ip = '0.0.0.0'
    assert is_ip_v4(zero_ip) == True, f"Expected True for valid IP {zero_ip}"

    # Test an IP address with a segment greater than 255 (should return False)
    beyond_range_ip = '192.168.1.300'
    assert is_ip_v4(beyond_range_ip) == False, f"Expected False for invalid IP {beyond_range_ip}"

    # Test an edge case where a segment is negative
    negative_segment_ip = '192.168.-1.1'
    assert is_ip_v4(negative_segment_ip) == False, f"Expected False for invalid IP {negative_segment_ip}"

    # Additional edge case: Test an IP with two segments out of range
    multiple_invalid_segments_ip = '300.256.1.1'
    assert is_ip_v4(multiple_invalid_segments_ip) == False, f"Expected False for invalid IP {multiple_invalid_segments_ip}"

    # Test an IP that is an empty string
    empty_ip = ''
    assert is_ip_v4(empty_ip) == False, f"Expected False for invalid IP {empty_ip}"

    # Non-IP string input
    non_ip_string = 'Hello, World!'
    assert is_ip_v4(non_ip_string) == False, f"Expected False for invalid IP {non_ip_string}"