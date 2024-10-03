from string_utils.validation import is_ip_v4

def test__is_ip_v4_mutant_killing():
    """
    This test checks the behavior of the is_ip_v4 function to ensure that the logic for validating
    IP addresses is correct. The test input '255.200.100.75' is a valid IP address and should
    return True, but the mutant is expected to return False for this input due to incorrect logic.
    Therefore, this test will pass on the baseline and fail on the mutant, killing the mutant.
    """
    # This input should return True for a valid IP address
    valid_ip = '255.200.100.75'
    
    # Check the expected behavior
    output = is_ip_v4(valid_ip)
    print(f"Is '{valid_ip}' a valid IP? {output}")
    assert output == True, "The mutant should have failed this test."