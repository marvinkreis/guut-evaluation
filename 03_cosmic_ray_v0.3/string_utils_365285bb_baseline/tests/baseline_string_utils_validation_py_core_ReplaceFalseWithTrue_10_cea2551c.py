from string_utils.validation import is_uuid

def test_is_uuid():
    # Test with a valid UUID (not in hex format)
    valid_uuid = '6f8aa2f9-686c-4ac3-8766-5712354a04cf'
    assert is_uuid(valid_uuid) == True  # It should return True
    
    # Test with a valid hex representation of a UUID (should return False with mutant)
    hex_uuid = '6f8aa2f9686c4ac387665712354a04cf'
    
    # This should pass in the correct code since the default allow_hex is False
    assert is_uuid(hex_uuid) == False  # It should return False when not allowing hex
    
    # This should pass both in the correct and mutant code when explicitly using allow_hex=True
    assert is_uuid(hex_uuid, allow_hex=True) == True  # Should return True if allow_hex is True