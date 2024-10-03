from string_utils.generation import secure_random_hex

def test__secure_random_hex():
    """
    Test the secure_random_hex function to verify behavior
    when byte_count is greater than 1. The baseline should return
    a valid hex string, while the mutant only allows byte_count = 1,
    leading to a ValueError for byte_count values greater than 1.
    """
    # Test with byte_count of 3, which should pass for baseline
    try:
        output_for_3_bytes = secure_random_hex(3)
        print(f"Output when byte_count = 3: {output_for_3_bytes}")
    except ValueError as e:
        print(f"ValueError when byte_count = 3: {e}")

    # Test with byte_count of 1, which should pass for both
    output_for_1_byte = secure_random_hex(1)
    print(f"Output when byte_count = 1: {output_for_1_byte}")
    
    # Assert that the output for 3 bytes is a valid hexadecimal string
    assert isinstance(output_for_3_bytes, str) and len(output_for_3_bytes) == 6