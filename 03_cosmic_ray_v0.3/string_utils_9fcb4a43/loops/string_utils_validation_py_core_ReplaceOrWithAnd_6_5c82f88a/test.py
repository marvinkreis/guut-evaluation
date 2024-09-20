from string_utils.validation import is_ip

def test__is_ip():
    """Changing 'or' to 'and' in is_ip would cause it not to accept valid IPs."""
    # Test with a valid IPv4 address
    output_ipv4 = is_ip('255.200.100.75')
    assert output_ipv4 == True, "is_ip should return True for valid IPv4 address"

    # Test with a valid IPv6 address
    output_ipv6 = is_ip('2001:db8:85a3:0000:0000:8a2e:370:7334')
    assert output_ipv6 == True, "is_ip should return True for valid IPv6 address"

    # Combined check to highlight difference
    output_combined = is_ip('255.200.100.75') and is_ip('2001:db8:85a3:0000:0000:8a2e:370:7334')
    assert output_combined == True, "is_ip should correctly check combined valid IPs"