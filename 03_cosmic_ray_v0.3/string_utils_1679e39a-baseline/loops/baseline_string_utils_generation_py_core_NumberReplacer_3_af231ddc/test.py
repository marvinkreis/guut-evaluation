from string_utils.generation import secure_random_hex

def test__secure_random_hex():
    """
    Test that secure_random_hex raises a ValueError when byte_count is set to 0. 
    The original function requires byte_count to be >= 1, whereas the mutant allows 
    0, leading to an incorrect behavior that does not raise an error, causing 
    potential security vulnerabilities when generating cryptographic strings.
    """
    try:
        secure_random_hex(0)
    except ValueError as e:
        assert str(e) == 'byte_count must be >= 1'
    else:
        assert False, "Expected ValueError not raised"