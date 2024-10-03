from string_utils.generation import uuid

def test__uuid_hex():
    """
    Test the uuid function to ensure it returns the correct output based on the as_hex parameter.
    When as_hex is set to False, the baseline should return a UUID string, while the mutant should yield the hex representation of the UUID.
    Therefore, we expect the test to pass in the baseline and fail in the mutant.
    """
    output = uuid(as_hex=False)
    assert isinstance(output, str) and '-' in output  # Check for UUID format