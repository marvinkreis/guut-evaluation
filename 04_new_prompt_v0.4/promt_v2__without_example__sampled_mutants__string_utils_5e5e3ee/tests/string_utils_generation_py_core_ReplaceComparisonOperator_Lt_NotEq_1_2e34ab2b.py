from string_utils.generation import secure_random_hex

def test__secure_random_hex_killing_mutant():
    """
    This test checks the behavior of secure_random_hex with byte_count=2. 
    The baseline should return a valid hexadecimal string, while the mutant 
    should throw a ValueError as it no longer accepts byte_count values other than 1.
    """
    output = secure_random_hex(2)
    assert isinstance(output, str) and len(output) == 4  # Expecting a 4 character long hex string