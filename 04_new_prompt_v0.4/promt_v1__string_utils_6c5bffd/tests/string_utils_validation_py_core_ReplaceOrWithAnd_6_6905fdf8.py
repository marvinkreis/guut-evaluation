from string_utils.validation import is_ip

def test__is_ip_valid_v4():
    """
    Test the `is_ip` function with a valid IPv4 address. The baseline should return True for valid IPv4 addresses,
    while the mutant will return False due to its change in logic.
    """
    input_string_v4 = '255.200.100.75'  # Valid V4 IP
    output = is_ip(input_string_v4)
    assert output is True, f"Expected True for valid IPv4, got {output}"

    input_string_v6 = '2001:db8:85a3:0000:0000:8a2e:370:7334'  # Valid V6 IP
    output = is_ip(input_string_v6)
    assert output is True, f"Expected True for valid IPv6, got {output}"
    
    input_string_both = '255.200.100.75 and 2001:db8:85a3:0000:0000:8a2e:370:7334'  # Invalid IP format
    output = is_ip(input_string_both)
    assert output is False, f"Expected False for invalid IP combination, got {output}"