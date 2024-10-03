from string_utils.generation import uuid

def test__uuid():
    """
    Test whether the uuid function correctly returns the UUID in string format when as_hex is set to False.
    The input here sets as_hex to False, which means we expect a UUID in the standard format. This will fail for 
    the mutant version of the code where the behavior is inverted and will return the hex representation instead.
    """
    output = uuid(as_hex=False)
    assert '-' in output  # A standard UUID string contains hyphens.