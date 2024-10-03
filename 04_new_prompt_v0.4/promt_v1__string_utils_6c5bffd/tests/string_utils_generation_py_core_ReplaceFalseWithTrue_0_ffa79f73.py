from string_utils.generation import uuid

def test__uuid_default():
    """
    Test that the uuid() function, when called with no arguments, produces a UUID in the default format,
    which contains dashes. The mutant changes the default behavior to return a hexadecimal UUID without dashes,
    which fails this assertion.
    """
    output = uuid()  # No arguments, should return a UUID in default format
    print(f"output = {output}")
    assert '-' in output  # Check for the presence of dashes (format indicator)