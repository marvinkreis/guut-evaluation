from string_utils._regex import JSON_WRAPPER_RE

def test__json_wrapper_regex():
    """Testing JSON_WRAPPER_RE regex to ensure it matches valid JSON strings."""
    
    valid_jsons = [
        '''{"key": "value"}''',
        '''{"name": "John", "age": 30, "car": null}''',
        '''{"students": [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 22}], "total": 2}'''
    ]
    
    for json_string in valid_jsons:
        match = JSON_WRAPPER_RE.match(json_string)
        assert match is not None, f"Failed to match valid JSON: {json_string}"