from string_utils._regex import HTML_RE

def test__html_re():
    # A sample HTML string for testing
    test_html = "<div>Hello World!</div>"
    
    # The original regex should match the test HTML string
    matched = HTML_RE.match(test_html)
    
    # Assert that the regex matches the string
    assert matched is not None, "The HTML_RE should match valid HTML tags."
    
    # Further check that the matched string is the same as input
    assert matched.group(0) == test_html, "The matched result should exactly equal the input."