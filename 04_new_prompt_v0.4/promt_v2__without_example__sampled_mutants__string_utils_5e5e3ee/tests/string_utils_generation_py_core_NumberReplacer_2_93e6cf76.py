from string_utils.generation import secure_random_hex

def test__secure_random_hex_kill_mutant():
    """
    Test the secure_random_hex function with byte_count of 1 which should succeed in the Baseline,
    but raise a ValueError in the Mutant. This test should clearly differentiate the Baseline from the Mutant.
    """
    output = secure_random_hex(1)  # This should succeed in the Baseline, but fail in the Mutant
    assert isinstance(output, str) and len(output) == 2  # Since byte_count of 1 results in length 2 hex string