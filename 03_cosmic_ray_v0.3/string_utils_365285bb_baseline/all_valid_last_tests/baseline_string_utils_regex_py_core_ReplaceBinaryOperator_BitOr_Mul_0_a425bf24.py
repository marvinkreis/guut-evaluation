from string_utils._regex import JSON_WRAPPER_RE

def test_json_wrapper_regex():
    valid_json_string = '[{"name": "John Doe", "age": 30}]'
    invalid_json_string = '[{"name": "John Doe", "age": 30'

    # Test that the valid JSON string matches the regex
    assert JSON_WRAPPER_RE.match(valid_json_string) is not None, "Valid JSON should match"

    # Test that the invalid JSON string does not match the regex
    assert JSON_WRAPPER_RE.match(invalid_json_string) is None, "Invalid JSON should not match"