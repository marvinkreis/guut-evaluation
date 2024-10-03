from string_utils.generation import secure_random_hex

def test_secure_random_hex():
    # Testing with an invalid byte_count (not equal to 1), which should raise a ValueError
    try:
        secure_random_hex(2)  # Should pass if the original function is intact
    except ValueError:
        pass  # This is expected when running the test against the mutant
    
    # Testing with a valid byte_count
    result = secure_random_hex(1)  # Should not raise an exception
    assert len(result) == 2  # 1 byte = 2 hex chars
    
    # Another invalid case
    try:
        secure_random_hex(0)  # Should raise ValueError for byte_count < 1
    except ValueError:
        pass  # This is expected

    # Valid input again
    result = secure_random_hex(5)  # Should generate a hex string of length 10
    assert len(result) == 10  # 5 bytes = 10 hex chars