from string_utils._regex import HTML_RE

def test__html_regex():
    # Test string containing valid HTML
    test_string = "<div>Hello World</div>"
    
    # Use the HTML_RE to check for a match
    match = HTML_RE.match(test_string)
    
    # Assert that a match is found with the correct regex
    assert match is not None, "HTML_RE should match valid HTML string"
    
    # String that is not a valid HTML to ensure it does not match incorrectly
    invalid_string = "Just a plain text without tags"
    
    # Check that the invalid string does not match
    invalid_match = HTML_RE.match(invalid_string)
    assert invalid_match is None, "HTML_RE should not match invalid HTML string"

test__html_regex()