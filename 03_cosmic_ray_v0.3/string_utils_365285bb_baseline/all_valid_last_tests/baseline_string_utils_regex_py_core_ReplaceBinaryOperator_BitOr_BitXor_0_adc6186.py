from string_utils._regex import JSON_WRAPPER_RE

def test__JSON_WRAPPER_RE():
    # Test case with well-formed multi-line JSON input
    valid_json = '''
    {
        "key": "value"
    }
    '''
    # Expect a match with the original regex
    assert JSON_WRAPPER_RE.match(valid_json) is not None, "The valid multi-line JSON should match the regex."

    # Test case with more complex valid JSON input (should still match)
    complex_json = '{ "key1": "value1", "key2": { "nestedKey": "nestedValue" } }'
    # Expect a match with the original regex
    assert JSON_WRAPPER_RE.match(complex_json) is not None, "The complex JSON should match the regex."
    
    # Test case with invalid JSON input (we expect it to not match)
    invalid_json = '{ key: "value" '  # Missing quote for key
    # Expect both original and mutant regex to not match
    assert JSON_WRAPPER_RE.match(invalid_json) is None, "The invalid JSON should not match the regex."