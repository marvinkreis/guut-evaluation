from string_utils.validation import is_json

def test_is_json():
    # This string is a valid JSON
    valid_json = '{"name": "Alice", "age": 30}'
    assert is_json(valid_json) == True, "Expected True for a valid JSON string"

    # This string is an invalid JSON
    invalid_json = '{name: "Alice", age: 30}'  # Missing quotes around keys
    assert is_json(invalid_json) == False, "Expected False for an invalid JSON string"

    # This will test an edge case with an empty JSON array
    empty_json_array = '[]'
    assert is_json(empty_json_array) == True, "Expected True for an empty JSON array"

    # This will test an edge case with an empty JSON object
    empty_json_object = '{}'
    assert is_json(empty_json_object) == True, "Expected True for an empty JSON object"