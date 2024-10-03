from string_utils.generation import secure_random_hex

def test_secure_random_hex_killing():
    """
    Test the secure_random_hex function using a byte count of 2. The mutant will raise
    a ValueError while the baseline will generate a valid hexadecimal string.
    """
    try:
        output = secure_random_hex(2)
        print(f"Output: {output}")  # Expecting a successful output from the baseline.
    except ValueError as ve:
        assert False, f"Expected successful output, but got ValueError: {ve}"