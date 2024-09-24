from string_utils.validation import is_ip_v6

def test_is_ip_v6():
    # Test with inputs that should return False
    assert not is_ip_v6(None)  # Should return False
    assert not is_ip_v6('')     # Should return False
    assert not is_ip_v6(' ')    # Should return False

    # This should be valid in the correct implementation
    assert is_ip_v6('2001:db8:85a3:0000:0000:8a2e:370:7334')  # Valid IP v6