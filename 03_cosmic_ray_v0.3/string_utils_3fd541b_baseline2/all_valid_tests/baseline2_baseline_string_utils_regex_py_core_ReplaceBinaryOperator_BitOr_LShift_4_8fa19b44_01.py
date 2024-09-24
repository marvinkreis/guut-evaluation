from string_utils._regex import HTML_RE

def test__html_regex_matching():
    # Input HTML string
    html_string = "<html><body><h1>Hello World!</h1></body></html>"
    
    # Test the regex matches the input HTML string
    match = HTML_RE.match(html_string)
    
    # Assert that we find a match
    assert match is not None, "Expected a match with the HTML string"