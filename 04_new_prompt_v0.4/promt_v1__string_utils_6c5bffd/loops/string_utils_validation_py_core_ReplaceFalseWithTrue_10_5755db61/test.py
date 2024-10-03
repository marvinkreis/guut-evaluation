from string_utils.validation import is_uuid

def test__is_uuid():
    """
    Test the is_uuid function to verify the impact of the allow_hex parameter.
    The input is a valid hex representation of a UUID. The original implementation should return False,
    while the mutant version should return True due to the changed default value of allow_hex.
    """
    # Expect the result to be False in the baseline and True in the mutant
    assert not is_uuid('6f8aa2f9686c4ac387665712354a04cf'), "Base uuid check failed"