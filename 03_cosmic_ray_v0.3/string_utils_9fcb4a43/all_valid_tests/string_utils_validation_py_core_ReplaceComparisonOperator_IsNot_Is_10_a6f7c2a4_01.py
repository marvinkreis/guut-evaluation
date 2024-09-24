from string_utils.validation import is_uuid

def test__is_uuid():
    """The mutant implementation incorrectly returns False for valid UUIDs and True for invalid ones."""
    
    valid_uuid = '6f8aa2f9-686c-4ac3-8766-5712354a04cf'
    invalid_uuid = '6f8aa2f9686c4ac387665712354a04cf'
    
    # Test valid UUID
    valid_output = is_uuid(valid_uuid)
    assert valid_output is True, "is_uuid must return True for valid UUIDs"
    
    # Test invalid UUID
    invalid_output = is_uuid(invalid_uuid)
    assert invalid_output is False, "is_uuid must return False for invalid UUIDs"