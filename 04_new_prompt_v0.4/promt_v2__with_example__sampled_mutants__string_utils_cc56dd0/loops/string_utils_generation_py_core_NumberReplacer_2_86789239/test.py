from string_utils.generation import secure_random_hex

def test_secure_random_hex_mutant_killing():
    """
    Test the secure_random_hex function using byte_count of 1.
    The baseline should generate a hex string, while the mutant will raise a ValueError
    indicating the byte_count must be >= 1.
    """
    try:
        output = secure_random_hex(1)
        assert isinstance(output, str) and len(output) == 2  # Valid hex string length for 1 byte
    except ValueError as ve:
        assert False, f"Expected a successful output, but raised ValueError: {ve}"