from string_utils.generation import uuid

def test_uuid_mutant_detection():
    # Generate UUID both ways
    standard_uuid = uuid(as_hex=False)
    hex_uuid = uuid(as_hex=True)

    # Ensure that the outputs are truly different, indicating the mutant failed to replicate this
    assert standard_uuid != hex_uuid, "The standard UUID and hex UUID should be different."

    # Check the standard UUID format
    assert len(standard_uuid) == 36, "The standard UUID should be 36 characters long with hyphens."
    assert standard_uuid.count('-') == 4, "Standard UUID should have 4 hyphens."
    assert all(c in '0123456789abcdef' for c in standard_uuid.replace('-', '')), "Standard UUID should only contain hex digits."

    # Check the hex UUID format
    assert len(hex_uuid) == 32, "The hex UUID should be 32 characters long without hyphens."
    assert all(c in '0123456789abcdef' for c in hex_uuid), "Hexadecimal UUID should contain only hex digits."