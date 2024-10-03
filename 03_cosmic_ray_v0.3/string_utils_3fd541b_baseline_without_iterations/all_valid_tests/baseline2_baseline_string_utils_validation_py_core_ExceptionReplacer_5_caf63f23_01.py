from string_utils.validation import is_json

def test__is_json():
    # This test case is designed to detect the mutant by ensuring that a valid JSON string
    # causes the function to return True, while an improper exception handling will cause it to fail.
    
    valid_json = '{"name": "Peter"}'
    invalid_json = '{nope}'  # invalid JSON string

    # Testing with a valid JSON string should return True
    assert is_json(valid_json) == True, "Expected True for valid JSON"

    # Testing with an invalid JSON string should return False
    assert is_json(invalid_json) == False, "Expected False for invalid JSON"

    # If the mutant is present and raises CosmicRayTestingException instead of the expected exceptions,
    # the function would likely not catch that error, leading to a failure in this case.
