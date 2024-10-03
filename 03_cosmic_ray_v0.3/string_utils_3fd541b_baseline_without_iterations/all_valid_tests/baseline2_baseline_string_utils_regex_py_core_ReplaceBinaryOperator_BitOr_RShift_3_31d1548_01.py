from string_utils._regex import HTML_RE

def test__html_regex():
    # Test case: A well-formed HTML string
    html_string = '<div class="example">Hello, World!</div>'
    
    # The regex should match the HTML string as valid
    assert HTML_RE.match(html_string) is not None, "The HTML regex should match valid HTML."
    
    # Test case: An invalid HTML string (missing closing tag)
    invalid_html_string = '<div class="example">Hello, World!'
    
    # The regex should also successfully match invalid HTML
    assert HTML_RE.match(invalid_html_string) is not None, "The HTML regex should still match potentially incomplete HTML."

# The test should pass when executed with the correct code
# and will fail with the mutant due to the improper use of `>>`.