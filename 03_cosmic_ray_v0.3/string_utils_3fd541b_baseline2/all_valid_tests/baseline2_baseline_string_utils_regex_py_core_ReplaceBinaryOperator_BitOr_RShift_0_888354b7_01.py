from string_utils._regex import JSON_WRAPPER_RE

def test__json_wrapper():
    # Case where input is a valid JSON string
    json_string = '{"key": "value"}'
    match = JSON_WRAPPER_RE.match(json_string)
    assert match is not None, "The regex should match valid JSON strings."

    # Case where input is not a valid JSON string
    invalid_json_string = 'key value'
    match_invalid = JSON_WRAPPER_RE.match(invalid_json_string)
    assert match_invalid is None, "The regex should not match invalid JSON strings."

    # Case where input has extra spaces but is still a valid JSON string
    json_string_with_spaces = '   [  { "key": "value" }  ]   '
    match_with_spaces = JSON_WRAPPER_RE.match(json_string_with_spaces)
    assert match_with_spaces is not None, "The regex should match valid JSON with extra spaces."