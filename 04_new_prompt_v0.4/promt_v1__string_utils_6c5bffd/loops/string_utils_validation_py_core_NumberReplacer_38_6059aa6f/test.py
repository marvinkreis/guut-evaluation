from string_utils.validation import is_ip_v4

def test__is_ip_v4_invalid_case():
    """
    Test to check the IP address validation for an out-of-range octet.
    The test input '192.168.1.256' is expected to return False for the baseline,
    indicating it is not a valid IPv4 address. The mutant should return True,
    thus differentiating between the two implementations.
    """
    input_string = "192.168.1.256"
    assert not is_ip_v4(input_string), "Expected output is False for this test case"