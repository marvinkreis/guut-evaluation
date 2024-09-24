from string_utils._regex import JSON_WRAPPER_RE

def test__json_wrapper_regex():
    # Valid JSON array
    valid_json_array = '[{"key": "value"}, {"key2": "value2"}]'
    assert JSON_WRAPPER_RE.match(valid_json_array) is not None, "The valid JSON array should match the regex."
    
    # Valid JSON object
    valid_json_object = '{"key": "value", "key2": "value2"}'
    assert JSON_WRAPPER_RE.match(valid_json_object) is not None, "The valid JSON object should match the regex."
    
    # Invalid JSON string
    invalid_json_string = 'just a string, not a json'
    assert JSON_WRAPPER_RE.match(invalid_json_string) is None, "The invalid JSON string should not match the regex."