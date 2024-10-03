from string_utils.validation import is_uuid

def test_is_uuid_mutant_killer():
    """
    Test to verify that the is_uuid function correctly identifies valid UUIDs.
    A valid UUID should return True in the baseline and False in the mutant due to the mutation.
    This test uses a known valid UUID.
    """
    valid_uuid = '6f8aa2f9-686c-4ac3-8766-5712354a04cf'
    
    baseline_result = is_uuid(valid_uuid)
    assert baseline_result == True, f"Expected True, got {baseline_result}"