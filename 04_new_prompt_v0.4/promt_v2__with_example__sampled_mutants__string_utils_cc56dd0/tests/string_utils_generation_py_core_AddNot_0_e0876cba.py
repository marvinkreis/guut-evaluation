from string_utils.generation import uuid

def test_uuid_mutant_killing():
    """
    Test the uuid function with as_hex=True. The baseline should return a
    hexadecimal string of length 32, while the mutant will return a string
    representation of UUID that includes dashes.
    """
    output = uuid(as_hex=True)
    assert len(output) == 32 and all(c in '0123456789abcdef' for c in output), "Expected output to be a hex string"