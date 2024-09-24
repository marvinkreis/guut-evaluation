from string_utils._regex import HTML_RE

def test__HTML_RE():
    # Using an input string that clearly should not be matched as HTML
    test_string = "Hello World without any HTML tags"

    # Try to match against the regex
    match = HTML_RE.match(test_string)
    
    # The correct implementation should return None since this is not valid HTML
    assert match is None, "The HTML regular expression should NOT match non-HTML strings."
    
    # Next, testing with a simple valid HTML string
    valid_html_string = "<p>Hello World</p>"
    valid_match = HTML_RE.match(valid_html_string)
    
    # We also need to ensure valid HTML matches
    assert valid_match is not None, "The HTML regular expression should match valid HTML."