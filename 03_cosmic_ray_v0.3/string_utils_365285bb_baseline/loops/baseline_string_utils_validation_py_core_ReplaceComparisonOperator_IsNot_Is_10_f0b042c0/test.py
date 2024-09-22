from string_utils.validation import is_uuid

def test_is_uuid():
    # This is a valid UUID
    valid_uuid = '6f8aa2f9-686c-4ac3-8766-5712354a04cf'
    assert is_uuid(valid_uuid) == True, "Expected True for valid UUID"

    # Edge case: Not a valid UUID
    invalid_uuid = 'invalid-uuid-string'
    assert is_uuid(invalid_uuid) == False, "Expected False for invalid UUID"

    # Additional valid UUID for testing
    another_valid_uuid = '123e4567-e89b-12d3-a456-426614174000'
    assert is_uuid(another_valid_uuid) == True, "Expected True for another valid UUID"