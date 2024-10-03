from string_utils.generation import uuid

def test_uuid_mutant_killing():
    """
    Test the uuid function by checking the default output format.
    The baseline should return a UUID string with dashes, while the mutant will only return a hex string.
    This confirms the mutant is detected because the expected format will not match.
    """
    output = uuid()
    assert '-' in output, "Expected output to contain dashes, but got: {}".format(output)