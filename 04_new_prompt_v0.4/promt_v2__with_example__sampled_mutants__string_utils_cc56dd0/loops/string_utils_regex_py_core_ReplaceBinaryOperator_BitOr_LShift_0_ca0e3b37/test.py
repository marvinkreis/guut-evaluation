from string_utils._regex import JSON_WRAPPER_RE

def test_json_wrapper_re_mutant_killing():
    """
    Test JSON_WRAPPER_RE with a well-structured JSON string containing spaces and line breaks.
    The baseline should return a match object, while the mutant should return None.
    This test case kills the mutant since it produces different results for the baseline.
    """
    spaced_json_string = '''
    {
        "key": {
            "inner_key": "value"
        }
    }
    '''
    output = JSON_WRAPPER_RE.match(spaced_json_string)
    assert output is not None, "Expected a match for spaced JSON, but received None."