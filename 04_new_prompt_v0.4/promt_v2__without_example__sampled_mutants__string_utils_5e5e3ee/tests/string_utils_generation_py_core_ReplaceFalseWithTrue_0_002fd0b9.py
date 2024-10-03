from string_utils.generation import uuid

def test__uuid():
    """
    Test the uuid function with default parameters.
    The baseline should return a UUID in the standard format, while the mutant should return a hex string.
    Specifically, the length of the output should be different.
    """
    output = uuid()
    is_baseline = len(output) == 36 and '-' in output  # UUID has dashes and is 36 characters long.
    assert is_baseline