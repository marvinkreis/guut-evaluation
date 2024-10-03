from string_utils.validation import is_uuid

def test_is_uuid_mutant_killing():
    """
    Test the is_uuid function with a valid hex UUID.
    The mutant should return True (due to allow_hex being True by default), 
    while the baseline should return False, indicating that hex UUIDs are not valid by default.
    """
    output = is_uuid('6f8aa2f9686c4ac387665712354a04cf')
    assert output == False, f"Expected False, got {output}"