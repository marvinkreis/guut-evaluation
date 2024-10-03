from string_utils._regex import JSON_WRAPPER_RE

def test__json_wrapper_regex():
    """
    Test that JSON_WRAPPER_RE correctly matches multi-line JSON strings.
    The input is a well-formed JSON string spanning multiple lines.
    This test reveals the difference in behavior between the baseline and mutant implementation
    of JSON_WRAPPER_RE, as the mutant fails to match the valid JSON.
    """
    json_string = """
    {
        "key": "value",
        "array": [
            1,
            2,
            3
        ]
    }
    """
    output = JSON_WRAPPER_RE.match(json_string)
    assert output is not None, "JSON_WRAPPER_RE should match well-formed JSON."