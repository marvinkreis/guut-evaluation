from string_utils.manipulation import roman_encode

def test__roman_encode_invalid_inputs():
    """Invalid inputs (0, negative numbers, and large numbers over 3999) should result in ValueError."""
    invalid_inputs = [0, -5, 4000]

    for val in invalid_inputs:
        try:
            roman_encode(val)
            assert False, f"Expected ValueError for input {val}, but no exception raised."
        except ValueError:
            # This is expected, so we pass the test
            pass