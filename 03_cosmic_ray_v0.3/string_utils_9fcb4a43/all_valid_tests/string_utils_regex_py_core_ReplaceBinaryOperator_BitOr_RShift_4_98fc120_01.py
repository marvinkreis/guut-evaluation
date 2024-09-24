from string_utils._regex import HTML_RE

def test__html_regex():
    """Testing the HTML_RE regex for edge cases with complex input that may expose issues in mutant implementations."""
    
    complex_malformed_html = "<div><p>Test <div>Nested <div></p></div></div>"
    
    # Testing with the correct implementation
    match = HTML_RE.match(complex_malformed_html)
    assert match is not None, "The HTML_RE should match complex malformed HTML."
    print(f"Matched groups: {match.groups()}")