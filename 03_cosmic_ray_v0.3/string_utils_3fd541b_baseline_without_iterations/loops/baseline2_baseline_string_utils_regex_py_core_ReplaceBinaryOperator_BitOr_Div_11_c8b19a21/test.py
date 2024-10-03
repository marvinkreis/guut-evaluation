from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    # Testing if the regex can still function as intended
    # by checking a string that should match the 'SPACES_INSIDE' rule

    test_string_match = '"Hello, World!"'
    test_string_non_match = 'Hello World!'

    # Should find the quoted string
    match = PRETTIFY_RE['SPACES_INSIDE'].search(test_string_match)
    assert match is not None, "The regex should match the quoted text."

    # Should not find any match
    no_match = PRETTIFY_RE['SPACES_INSIDE'].search(test_string_non_match)
    assert no_match is None, "The regex should NOT match the non-quoted text."

    # Additional test: string inside brackets
    test_string_bracket = '(Sample text)'
    match_bracket = PRETTIFY_RE['SPACES_INSIDE'].search(test_string_bracket)
    assert match_bracket is not None, "The regex should match the text inside brackets."

    # Additional test on a string with multiple spaces inside quotes
    test_string_multiple_spaces = '"This   is an example."'
    match_multiple_spaces = PRETTIFY_RE['SPACES_INSIDE'].search(test_string_multiple_spaces)
    assert match_multiple_spaces is not None, "The regex should match the quoted text with multiple spaces."