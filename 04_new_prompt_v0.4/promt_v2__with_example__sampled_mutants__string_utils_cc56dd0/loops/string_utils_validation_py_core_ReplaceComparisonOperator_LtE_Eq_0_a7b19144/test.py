from string_utils.validation import is_ip_v4

def test_is_ip_v4_mutant_killing():
    """
    Test the is_ip_v4 function using the valid IPv4 address '0.0.0.1'.
    The baseline will return True indicating a valid IP address, while the mutant
    will incorrectly return False due to its altered condition, thus killing the mutant.
    """
    output = is_ip_v4("0.0.0.1")
    assert output == True, f"Expected True, got {output}"