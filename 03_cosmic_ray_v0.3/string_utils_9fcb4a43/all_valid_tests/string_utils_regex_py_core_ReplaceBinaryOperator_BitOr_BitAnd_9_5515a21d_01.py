from string_utils._regex import PRETTIFY_RE

def test__uppercase_after_sign():
    """Ensure uppercase letters after punctuation match correctly."""
    input_strings = [
        "Hello! How are you? This is a test.",  # Should match: ['! H', '? T']
        "WOW! You did it!! Great job?",           # Should match: ['! Y', '! G']
        "@A start of a line, followed by more!",  # Should NOT match
        "#1 something here? Yes!"                  # Should match: ['? Y']
    ]
    
    expected_outputs = [
        ['! H', '? T'],
        ['! Y', '! G'],
        [],
        ['? Y']
    ]
    
    for test_string, expected in zip(input_strings, expected_outputs):
        actual_matches = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].findall(test_string)
        assert actual_matches == expected, f"Expected: {expected}, but got: {actual_matches}"
