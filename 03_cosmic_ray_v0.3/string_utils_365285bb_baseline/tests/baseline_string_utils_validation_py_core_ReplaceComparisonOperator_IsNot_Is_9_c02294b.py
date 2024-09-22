from string_utils.validation import is_uuid

def test__is_uuid():
    # Test for a valid UUID (standard format)
    valid_uuid = '6f8aa2f9-686c-4ac3-8766-5712354a04cf'
    assert is_uuid(valid_uuid) == True  # should return True in the original code
    assert is_uuid(valid_uuid, allow_hex=True) == True  # should also return True if allow_hex is True

    # Test for an invalid UUID
    invalid_uuid = 'not-a-uuid'
    assert is_uuid(invalid_uuid) == False  # should return False

    # Test for valid UUID hex representation
    valid_hex_uuid = '6f8aa2f9686c4ac387665712354a04cf'
    assert is_uuid(valid_hex_uuid, allow_hex=True) == True  # should return True in the original code

    print("All assertions passed!")

# Uncomment below line to run the test case manually
# test__is_uuid()