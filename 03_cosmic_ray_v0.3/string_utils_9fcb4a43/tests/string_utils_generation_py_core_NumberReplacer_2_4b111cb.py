from string_utils.generation import secure_random_hex

def test__secure_random_hex():
    """Changing the condition for byte_count to be >= 2 in secure_random_hex will raise ValueError for byte_count = 1."""
    try:
        output = secure_random_hex(1)
        assert output is not None, "secure_random_hex must produce a valid string for byte_count = 1"
    except ValueError:
        assert False, "secure_random_hex should not raise ValueError for byte_count = 1"