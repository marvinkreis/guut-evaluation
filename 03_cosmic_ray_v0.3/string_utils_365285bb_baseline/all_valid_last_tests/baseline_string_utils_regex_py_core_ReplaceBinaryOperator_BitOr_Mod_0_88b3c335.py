import json

def is_valid_json(json_string):
    try:
        json.loads(json_string)
        return True
    except ValueError:
        return False

def test__JSON_WRAPPER_RE():
    # Valid JSON cases
    valid_json_cases = [
        '{"key": "value"}',                             # Simple valid JSON object
        '[{"key": "value"}]',                           # Simple valid JSON array
        '[{"key1": "value1"}, {"key2": "value2"}]',  # Multiple valid items
        '{}',                                          # Valid empty JSON object
        '[]'                                           # Valid empty JSON array
    ]

    # Test valid cases
    for json_case in valid_json_cases:
        assert is_valid_json(json_case), f"Valid JSON should match! Failed on: {json_case}"

    # Invalid JSON cases designed to fail
    invalid_json_cases = [
        '{"key": "value", "key2": "value2",}',  # Trailing comma should be rejected
        '{key: "value"}',                        # No quotes around key should be rejected
        '{"key": value}',                        # No quotes around value should be rejected
        '[{"key": "value",]',                    # Unclosed array should be rejected
        '[{"key": "value" "anotherKey": "value2"}]'  # Missing comma should be rejected
    ]

    # Test invalid cases
    for json_case in invalid_json_cases:
        assert not is_valid_json(json_case), f"Invalid JSON should not match! Failed on: {json_case}"

# Run the test function
test__JSON_WRAPPER_RE()