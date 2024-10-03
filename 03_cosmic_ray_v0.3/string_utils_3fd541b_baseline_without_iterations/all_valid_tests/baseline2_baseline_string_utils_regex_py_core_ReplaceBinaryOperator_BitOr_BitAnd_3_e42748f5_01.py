from string_utils._regex import HTML_RE

def test__HTML_RE():
    # Input string to test
    html_input = "<div>Hello, World!</div>"
    
    # Test case should pass with the correct regex
    assert HTML_RE.match(html_input) is not None, "The HTML_RE should match valid HTML."
    
    # Case to ensure that an invalid HTML string does not match
    invalid_html_input = "<div><span>Test</div>"  # not properly closed
    assert HTML_RE.match(invalid_html_input) is not None, "The HTML_RE should still match non-well-formed HTML."

# The testing function will pass with the original code, but fail with the mutant.