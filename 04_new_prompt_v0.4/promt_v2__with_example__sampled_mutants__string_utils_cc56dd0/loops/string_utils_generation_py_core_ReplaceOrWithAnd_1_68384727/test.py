from string_utils.generation import secure_random_hex

def test_secure_random_hex_kills_mutant():
    """
    Test the secure_random_hex function with an invalid byte_count.
    The baseline will raise a ValueError for byte_count = 0,
    while the mutant will not raise any exception.
    """
    try:
        secure_random_hex(0)
        assert False, "Expected ValueError not raised."
    except ValueError as e:
        assert str(e) == "byte_count must be >= 1", f"Unexpected error message: {e}"