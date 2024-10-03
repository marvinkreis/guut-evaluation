from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_re():
    # Test string with a valid HTML tag
    test_string = '<div></div>'
    
    # Using the HTML_TAG_ONLY_RE to find a match
    match = HTML_TAG_ONLY_RE.match(test_string)
    
    # Assertion to check if the match is successful
    assert match is not None, "The regex should match a valid HTML tag."

    # Test string with an invalid HTML tag
    invalid_test_string = 'not_a_tag'
    
    # Check that it does not match an invalid HTML string
    invalid_match = HTML_TAG_ONLY_RE.match(invalid_test_string)
    
    # This should remain None
    assert invalid_match is None, "The regex should not match an invalid HTML string."