from string_utils.validation import is_ip_v4

def test__is_ip_v4():
    """
    Test if the function correctly identifies that '256.100.100.1' is not a valid IPv4 address because one octet exceeds the maximum value of 255.
    The mutant incorrectly allows values of 256, so this test will succeed in the correct code and fail in the mutant.
    """
    output = is_ip_v4('256.100.100.1')
    assert output == False