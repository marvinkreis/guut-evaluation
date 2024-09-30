from string_utils.validation import is_ip_v6

def test__is_ip_v6():
    """
    Test whether the function properly validates an IPv6 address. The input is a valid IPv6 address
    '2001:db8:85a3:0000:0000:8a2e:370:7334', which should return True. The mutant changes the logic 
    to use 'or' instead of 'and', causing the function to return True even for invalid strings. 
    Therefore, we will verify against an invalid input 'invalid_ip_address' which should return 
    False in the baseline but True in the mutant, thus detecting the mutant.
    """
    # Test with a valid IPv6 address
    valid_output = is_ip_v6('2001:db8:85a3:0000:0000:8a2e:370:7334')
    assert valid_output == True

    # Test with an invalid input to detect mutant behavior
    invalid_output = is_ip_v6('invalid_ip_address')
    assert invalid_output == False