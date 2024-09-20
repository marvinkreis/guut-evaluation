from string_utils.validation import is_uuid

def test__is_uuid():
    """The mutant changes the UUID validation logic, causing it to incorrectly reject valid hex UUIDs."""
    valid_hex_uuid = '6f8aa2f9686c4ac387665712354a04cf'
    assert is_uuid(valid_hex_uuid, allow_hex=True) == True, "is_uuid should return True for valid hex UUID"