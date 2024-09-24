from string_utils.validation import is_ip_v4

def test__is_ip_v4():
    """The mutant code incorrectly checks for valid IPv4 addresses."""
    assert is_ip_v4("192.0.2.0") == True, "192.0.2.0 should be valid"
    assert is_ip_v4("255.255.255.255") == True, "255.255.255.255 should be valid"
    assert is_ip_v4("192.0.2.256") == False, "192.0.2.256 should be invalid"
    assert is_ip_v4("192.0.2.-1") == False, "192.0.2.-1 should be invalid"