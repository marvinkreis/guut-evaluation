from string_utils.validation import is_uuid

def test__is_uuid_mutant_kill():
    """
    Test whether the is_uuid function distinguishes correctly
    between valid and invalid UUIDs, especially when allow_hex is toggled.
    I expect that a valid hexadecimal UUID should return True on the baseline 
    and False on the mutant, which verifies the mutant's faulty logic.
    """
    valid_hex_uuid = '6f8aa2f9686c4ac387665712354a04cf'
    assert is_uuid(valid_hex_uuid, allow_hex=True) == True  # Should pass in baseline