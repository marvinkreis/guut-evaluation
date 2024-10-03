from string_utils.validation import is_ip_v6

def test_is_ip_v6_mutant_killing():
    """
    Test the is_ip_v6 function using a valid IPv6 address. The mutant will return False due to the inverted logic,
    while the baseline will correctly return True for valid inputs.
    """
    output = is_ip_v6('2001:db8:85a3:0000:0000:8a2e:370:7334')
    assert output == True, f"Expected True, got {output}"