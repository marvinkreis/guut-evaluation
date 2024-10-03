from string_utils.validation import is_uuid

def test__is_uuid_hex():
    """
    Test the is_uuid function for validity of hex UUIDs. The input '6f8aa2f9686c4ac387665712354a04cf' 
    is a valid hex UUID. The baseline should return True while the mutant should return False, thus 
    killing the mutant.
    """
    valid_hex_uuid = '6f8aa2f9686c4ac387665712354a04cf'  # valid hex UUID
    output = is_uuid(valid_hex_uuid, allow_hex=True)
    assert output is True, f"Expected True but got {output}"