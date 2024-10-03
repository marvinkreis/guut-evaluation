from string_utils.validation import is_ip_v6

def test__is_ip_v6_mutant_kill():
    """
    Test to check the behavior of is_ip_v6 function. 
    The function should return True for valid IPv6 addresses and False for invalid addresses.
    The test checks both valid and invalid IPv6 addresses to differentiate between the Baseline and the Mutant.
    """
    # Test valid IPv6
    valid_ip = '2001:db8:85a3:0000:0000:8a2e:370:7334'
    assert is_ip_v6(valid_ip) == True, "Should be True for valid IPv6"

    # Test invalid IPv6
    invalid_ip = '2001:db8:85a3:0000:0000:8a2e:370:?'
    assert is_ip_v6(invalid_ip) == False, "Should be False for invalid IPv6"