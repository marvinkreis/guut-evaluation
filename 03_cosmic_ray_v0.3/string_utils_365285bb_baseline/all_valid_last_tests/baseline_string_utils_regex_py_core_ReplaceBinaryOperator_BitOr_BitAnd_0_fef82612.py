from string_utils._regex import JSON_WRAPPER_RE

def test_JSON_WRAPPER_RE():
    # This JSON string should match the regex defined in JSON_WRAPPER_RE
    json_input_valid = """
    [
        {
            "key": "value"
        }
    ]
    """
    
    # This string should not match the regex as it's not a proper JSON structure
    json_input_invalid = "Not a JSON string"

    # Test that the regex matches correctly with the original regular expression
    assert JSON_WRAPPER_RE.match(json_input_valid) is not None, "Test case failed: Expected to match JSON format."
    # Test that the regex does not match the invalid JSON structure
    assert JSON_WRAPPER_RE.match(json_input_invalid) is None, "Test case failed: Expected NOT to match a non-JSON format."

    # If the mutant is applied, the assertions should fail so we detect the mutant