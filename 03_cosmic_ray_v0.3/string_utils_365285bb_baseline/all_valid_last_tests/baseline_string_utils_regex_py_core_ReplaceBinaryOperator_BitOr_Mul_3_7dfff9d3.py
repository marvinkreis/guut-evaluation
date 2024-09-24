from string_utils._regex import HTML_RE

def test__detect_html_re_mutant():
    # A simple HTML string
    test_string = "<html><body><h1>Hello World!</h1></body></html>"
    
    # Expect the regex to match the HTML string
    match = HTML_RE.match(test_string)
    
    # Assert that the match is successful 
    assert match is not None, "The regex should match valid HTML but it did not."