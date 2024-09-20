from string_utils.generation import secure_random_hex

def test__secure_random_hex():
    """Mutant changed the input validation, allowing byte_count=0, which should raise a ValueError."""
    try:
        secure_random_hex(0)
    except ValueError as e:
        assert str(e) == 'byte_count must be >= 1', "Expected ValueError not raised with correct message."
    else:
        assert False, "Expected ValueError when byte_count=0"