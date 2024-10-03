from string_utils.validation import is_ip_v4

def test__is_ip_v4_mutant_killing():
    """
    Test the is_ip_v4 function to confirm correct validation of IP addresses.
    The test will utilize specific invalid IP addresses which should yield
    'False' for the baseline while the mutant would incorrectly yield 'True'.
    """
    # Valid IP addresses
    assert is_ip_v4("192.168.0.1") == True  # Valid IP, expected True
    assert is_ip_v4("255.255.255.255") == True  # Valid IP, expected True

    # Invalid IP cases that should return False in the baseline but should return True in the mutant.
    assert is_ip_v4("") == False   # Expected to be False in baseline
    assert is_ip_v4("invalid_ip") == False  # Expected to be False in baseline
    assert is_ip_v4("256.100.50.25") == False  # Expected to be False in baseline
    assert is_ip_v4("300.100.50.25") == False  # Expected to be False in baseline