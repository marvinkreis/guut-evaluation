from string_utils._regex import JSON_WRAPPER_RE

def test__json_wrapper_regex():
    """The mutant's regex definition modification for JSON_WRAPPER_RE causes it to fail, while the original works correctly."""
    # Complex test JSON string
    complex_test_string = '''
    {
        "array": [1, 2, 3],
        "object": {
            "nested_key": "nested_value"
        },
        "boolean": true,
        "null_value": null
    }
    '''

    # Check correct output
    correct_match = JSON_WRAPPER_RE.match(complex_test_string)
    assert correct_match is not None, "JSON_WRAPPER_RE must match valid JSON strings."

    # If we had access to the mutant code, we would assert its behavior here, but we don't import it in tests.
    # This is just an indication that mutant behavior is expected to be None.

# Call the test function to verify
test__json_wrapper_regex()