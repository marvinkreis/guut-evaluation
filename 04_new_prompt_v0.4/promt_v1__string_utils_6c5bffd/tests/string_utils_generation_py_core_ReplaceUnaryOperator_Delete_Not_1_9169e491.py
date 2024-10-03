from string_utils.generation import secure_random_hex

def test__secure_random_hex_invalid_inputs():
    """
    Test the secure_random_hex function with invalid inputs to ensure it raises ValueError. This serves to
    confirm that only valid inputs (positive integers) are accepted. The mutant fails to catch a TypeError
    for non-integer inputs, while the baseline correctly raises a ValueError.
    """

    # Test with a negative integer
    try:
        secure_random_hex(-5)
        assert False, "ValueError expected for negative input, but none was raised"
    except ValueError as e:
        assert str(e) == 'byte_count must be >= 1'

    # Test with a non-integer input
    try:
        secure_random_hex('abc')
        assert False, "ValueError expected for non-integer input, but none was raised"
    except ValueError as e:
        assert str(e) == 'byte_count must be >= 1'  # Expecting ValueError as per baseline

    except TypeError as e:
        assert False, f"Unexpected TypeError: {e}"