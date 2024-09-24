from string_utils._regex import JSON_WRAPPER_RE

def test__json_wrapper_regex_edge_cases():
    """Testing JSON_WRAPPER_RE regex to recognize edge cases in valid JSON strings."""
    
    edge_cases = [
        '''{"name": "John \"Doe\""}''',  # Escape characters in value
        '''{"array": [1, 2, 3, 4]}''',     # Array within JSON
        '''{"object": {"inner": "value"}}''',  # Nested object
        '''[{"key": "value"}, {"key2": "value2"}]'''  # Top-level array
    ]
    
    for json_string in edge_cases:
        match = JSON_WRAPPER_RE.match(json_string)
        assert match is not None, f"Failed to match edge case valid JSON: {json_string}"