from string_utils.validation import is_json

def test__is_json():
    # Test input that is not valid JSON but is a non-empty string
    invalid_json_input = "just_a_string"
    
    # The correct code should return False, but the mutant would return True
    assert is_json(invalid_json_input) == False, "Expected False for invalid JSON input"