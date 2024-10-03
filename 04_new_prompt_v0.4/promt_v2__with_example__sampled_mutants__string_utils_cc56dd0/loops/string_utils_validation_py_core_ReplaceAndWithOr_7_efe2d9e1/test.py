from string_utils.validation import is_ip_v6

def test_is_ip_v6_mutant_killing():
    """
    Test the is_ip_v6 function with an invalid IPv6 address. The mutant will incorrectly validate it as true,
    while the baseline will correctly return false.
    """
    output = is_ip_v6('2001:0db8:85a3:0000:0000:8a2e:0370:7334::')
    assert output == False, f"Expected False, got {output}"