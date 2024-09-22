from string_utils.generation import secure_random_hex

def test_secure_random_hex():
    # This should raise a ValueError because byte_count is 0
    try:
        secure_random_hex(0)
        assert False, "Expected ValueError for byte_count = 0"
    except ValueError:
        pass  # Expected behavior
    
    # This should also raise a ValueError because byte_count is negative
    try:
        secure_random_hex(-1)
        assert False, "Expected ValueError for byte_count = -1"
    except ValueError:
        pass  # Expected behavior
    
    # Ensure that the function works correctly with valid input
    result = secure_random_hex(1)
    assert isinstance(result, str) and len(result) == 2, "Expected a valid hex string of length 2 for byte_count = 1"

    # A valid call that should return a hex string of length 10
    result = secure_random_hex(5)
    assert isinstance(result, str) and len(result) == 10, "Expected a valid hex string of length 10 for byte_count = 5"