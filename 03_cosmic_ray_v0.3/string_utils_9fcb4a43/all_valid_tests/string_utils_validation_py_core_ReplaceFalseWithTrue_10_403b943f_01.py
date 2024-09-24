from string_utils.validation import is_uuid

def test__is_uuid():
    """The mutant allows hex strings as valid UUIDs when allow_hex=True, while the original should not when allow_hex=False."""
    valid_uuid = '6f8aa2f9-686c-4ac3-8766-5712354a04cf'
    hex_uuid = '6f8aa2f9686c4ac387665712354a04cf'

    # assert True for a valid UUID
    assert is_uuid(valid_uuid, allow_hex=False) is True, "Valid UUID should return True"
    
    # assert False for a hex UUID when allow_hex is False
    assert is_uuid(hex_uuid, allow_hex=False) is False, "Hex UUID should return False when allow_hex=False"

    # assert True for hex UUID when allow_hex is True
    assert is_uuid(hex_uuid, allow_hex=True) is True, "Hex UUID should return True when allow_hex=True"