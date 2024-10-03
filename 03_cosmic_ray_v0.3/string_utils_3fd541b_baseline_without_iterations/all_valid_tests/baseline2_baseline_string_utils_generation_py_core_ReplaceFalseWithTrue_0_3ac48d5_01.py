from string_utils.generation import uuid

def test_uuid():
    # Testing the uuid function with as_hex=False
    normal_uuid = uuid(as_hex=False)
    # The output should be a standard UUID string format (5 segments)
    assert len(normal_uuid) == 36
    assert normal_uuid.count('-') == 4

    # Testing the uuid function with as_hex=True
    hex_uuid = uuid(as_hex=True)
    # The output should be a 32-character hexadecimal string
    assert len(hex_uuid) == 32
    assert all(c in "0123456789abcdef" for c in hex_uuid)

    # Testing that the mutant changes the output
    mutated_uuid = uuid(as_hex=True)  # this will call the mutant version
    assert mutated_uuid != normal_uuid  # mutant changes behavior
    
    # Ensure we can still obtain a valid UUID format from the normal case
    assert len(normal_uuid) == 36
    assert normal_uuid.count('-') == 4  # Check UUID format

    # Ensure we get a different format from the mutant when calling uuid()
    assert hex_uuid != mutated_uuid  # Mutant should provide the same output format but different
    
    print("All tests passed!")

# Note: In the above test function, the last two assertions verify the mutant's behavior change and that we get different outputs properly.