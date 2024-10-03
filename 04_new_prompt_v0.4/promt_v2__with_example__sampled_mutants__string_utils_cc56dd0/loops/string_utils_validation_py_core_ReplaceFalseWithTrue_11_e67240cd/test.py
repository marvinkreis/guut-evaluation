from string_utils.validation import is_ip_v4

def test_is_ip_v4_mutant_killing():
    """
    Test the is_ip_v4 function using None and an invalid IP address (256.100.200.75).
    The mutant will return True for None due to logic reversal, while the baseline will return False.
    """
    output_none = is_ip_v4(None)
    output_invalid_ip = is_ip_v4('256.100.200.75')
    
    assert output_none is False, f"Expected False for None, got {output_none}"
    assert output_invalid_ip is False, f"Expected False for 256.100.200.75, got {output_invalid_ip}"