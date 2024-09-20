from string_utils.generation import uuid

def test__uuid():
    """Changing the default `as_hex` parameter to True in uuid() alters the output format."""
    output = uuid()  # should return a standard UUID string
    assert '-' in output, "Expected output should contain dashes for a valid UUID"