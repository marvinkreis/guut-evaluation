from string_utils._regex import HTML_RE

def test__html_re():
    # A sample HTML string that we expect to be matched by the HTML_RE.
    html_string = "<div>Hello, World!</div>"
    
    # Check if the regex matches the HTML string
    match = HTML_RE.match(html_string)
    
    # Assert that the match is successful when using the original regex
    assert match is not None, "The HTML_RE should match a simple HTML string."
    
    # Check with a string that has invalid HTML, which should not match
    invalid_html_string = "Just some text without HTML."
    non_match = HTML_RE.match(invalid_html_string)
    
    # Assert that the HTML_RE does not match a non-HTML string
    assert non_match is None, "The HTML_RE should not match a string without HTML."