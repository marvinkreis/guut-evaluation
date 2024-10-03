from string_utils._regex import HTML_RE

def test__html_re_compilation():
    """
    Test the HTML_RE regex pattern to verify if it compiles successfully and matches a simple HTML string.
    This will help verify that the mutant introduces a syntax error in the regex due to incorrect operator usage,
    which will cause the test to fail when executed on the mutant.
    """
    html_string = "<div>Hello World</div>"
    # Check if the regex compiles and can match a valid HTML string
    result = HTML_RE.match(html_string)
    assert result is not None, "HTML_RE should match a valid HTML string."