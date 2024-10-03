from string_utils.validation import is_uuid

def test__is_uuid():
    # Test valid UUID
    valid_uuid = '6f8aa2f9-686c-4ac3-8766-5712354a04cf'
    assert is_uuid(valid_uuid) == True, "Expected valid UUID to return True"

    # Test invalid UUID
    invalid_uuid = '6f8aa2f9686c4ac387665712354a04cf'  # Not in the proper format
    assert is_uuid(invalid_uuid) == False, "Expected invalid UUID to return False"
    
    # Test edge case: empty string
    assert is_uuid('') == False, "Expected empty string to return False"
    
    # Test edge case: None input
    assert is_uuid(None) == False, "Expected None to return False"

    # Test edge case: string that is not a UUID
    non_uuid_string = 'not-a-uuid-string'
    assert is_uuid(non_uuid_string) == False, "Expected non-UUID string to return False"