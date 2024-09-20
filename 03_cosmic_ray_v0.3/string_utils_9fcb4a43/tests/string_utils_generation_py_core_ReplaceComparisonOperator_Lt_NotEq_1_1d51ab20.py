from string_utils.generation import secure_random_hex

def test__secure_random_hex():
    """The mutant change in secure_random_hex causes it to fail with byte_count of more than 1."""
    
    # Testing with valid large input
    output = secure_random_hex(1000)
    assert len(output) == 2000, "secure_random_hex should produce a hexadecimal string of length 2 * byte_count"
    
    # Testing with input of 1
    output_one = secure_random_hex(1)
    assert len(output_one) == 2, "secure_random_hex should produce a hexadecimal string of length 2 * byte_count for byte_count of 1"