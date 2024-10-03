from string_utils.validation import is_uuid

def test__is_uuid():
    # Test with a valid UUID
    valid_uuid = '6f8aa2f9-686c-4ac3-8766-5712354a04cf'
    assert is_uuid(valid_uuid) == True  # Should return True for a valid UUID

    # Test with a valid hexadecimal UUID
    valid_hex_uuid = '6f8aa2f9686c4ac387665712354a04cf'
    assert is_uuid(valid_hex_uuid, allow_hex=True) == True  # Should return True for valid hex UUID

    # Test with an invalid hex UUID format
    invalid_hex_uuid = '6f8aa2f9686c4ac387665712354a04cg'  # 'g' at the end makes it invalid
    assert is_uuid(invalid_hex_uuid, allow_hex=True) == False  # Should return False

    # Test with an invalid UUID
    invalid_uuid = 'invalid-uuid-string'
    assert is_uuid(invalid_uuid) == False  # Should return False for invalid UUID

    print("All assertions passed.")