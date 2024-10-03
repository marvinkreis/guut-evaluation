from string_utils._regex import JSON_WRAPPER_RE

def test__json_wrapper_newline():
    """
    This test checks whether the regex can match JSON strings containing
    newline characters. The baseline should match such strings, while the
    mutant is expected to fail.
    """
    valid_json_with_newline = '[{ \n "key": "value" \n }]'
    match = JSON_WRAPPER_RE.match(valid_json_with_newline)
    
    assert match is not None  # Expect a match with the baseline