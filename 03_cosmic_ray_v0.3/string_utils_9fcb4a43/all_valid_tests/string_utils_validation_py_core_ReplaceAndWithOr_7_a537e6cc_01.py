from string_utils.validation import is_ip_v6

def test__is_ip_v6():
    """The mutant changes the behavior of is_ip_v6, causing it to incorrectly accept invalid IPs."""
    assert not is_ip_v6(""), "An empty string should not be a valid IPv6 address."
    assert is_ip_v6("2001:db8:85a3:0000:0000:8a2e:370:7334"), "This is a valid IPv6 address."
    assert not is_ip_v6("invalid_ip"), "This should not be accepted as a valid IPv6 address."