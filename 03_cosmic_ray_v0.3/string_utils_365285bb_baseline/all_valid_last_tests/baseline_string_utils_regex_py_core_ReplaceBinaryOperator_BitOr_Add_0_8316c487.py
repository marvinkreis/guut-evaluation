from string_utils._regex import JSON_WRAPPER_RE

def test_JSON_WRAPPER_RE():
    # Valid JSON strings
    json_valid = [
        '{"key": "value"}',                 # Simple valid JSON
        '[{"key": "value"}]',                # Array with one object
        '[1, 2, 3]',                          # Array of numbers
        '{"array": [1, 2, 3]}',              # Valid JSON object containing an array
        '{"key": "value", "key2": "value2"}'# Valid JSON with multiple keys
    ]

    # Invalid JSON strings
    json_invalid = [
        '{key: value}',          # Missing quotes around key
        '[{"key": "value",}]',   # Trailing comma
        '{"key": "value" ',      # No closing brace
        '["key": "value"]',      # Incorrect structure
        'not a json string'      # Not JSON format
    ]
    
    # Check for valid cases
    for json_str in json_valid:
        match = JSON_WRAPPER_RE.match(json_str)
        assert match is not None, f"Valid JSON string failed: {json_str}"

    # Check for one typical edge case that hits the mutant while correct regex will accept it.
    json_edge_case = '[]\n  {"key": "value"}\n'  # JSON array with a newline before the object
    match_edge_case = JSON_WRAPPER_RE.match(json_edge_case)
    
    # Correct code should match the string
    assert match_edge_case is not None, f"Valid edge case failed: {json_edge_case}"

    # Test a well-formed JSON string with newline
    json_newline_case = '{\n  "key": "value"\n}\n'
    match_newline_case = JSON_WRAPPER_RE.match(json_newline_case)
    assert match_newline_case is not None, f"Edge case JSON failed to match: {json_newline_case}"

    # Now let's introduce an obviously incompatible string that we'd expect to fail.
    incompatible_string = '{"key": "value",\n'  # Missing closing brackets and thus should fail
    match_incompatible = JSON_WRAPPER_RE.match(incompatible_string)
    assert match_incompatible is None, f"Invalid JSON string matched: {incompatible_string}"

# To run the test
# test_JSON_WRAPPER_RE()