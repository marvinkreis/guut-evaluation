from string_utils.generation import secure_random_hex

def test_secure_random_hex():
    # This should pass since 1 is a valid input
    result = secure_random_hex(1)
    assert len(result) == 2, "Expected output length for byte_count 1 is 2 characters."
    
    # This should raise a ValueError since byte_count is 0
    try:
        secure_random_hex(0)
        assert False, "Expected ValueError for byte_count of 0"
    except ValueError:
        pass  # Correct behavior
    
    # This should raise a ValueError since byte_count is negative
    try:
        secure_random_hex(-1)
        assert False, "Expected ValueError for negative byte_count"
    except ValueError:
        pass  # Correct behavior

    # This should also raise a ValueError since byte_count is 1
    try:
        secure_random_hex(0)
        assert False, "Expected ValueError for byte_count of 0"
    except ValueError:
        pass  # Correct behavior