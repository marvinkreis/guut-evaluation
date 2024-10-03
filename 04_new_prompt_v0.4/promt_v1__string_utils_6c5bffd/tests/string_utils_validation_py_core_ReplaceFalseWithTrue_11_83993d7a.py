from string_utils.validation import is_ip_v4

def test__is_ip_v4_invalid_string():
    """
    Test whether the function correctly identifies a non-IP address string. 
    The input 'some.non.ip.string' should return False in the baseline
    but True in the mutant due to the change in the return statement, 
    allowing invalid IPs to be considered valid. This test effectively
    kills the mutant.
    """
    output = is_ip_v4("some.non.ip.string")
    assert output == False  # Expect False for baseline, should fail for mutant