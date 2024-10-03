from string_utils._regex import HTML_RE

def test_html_regex_mutant_killing():
    """
    Test the HTML_RE regex with a multi-line HTML string.
    The baseline will match successfully, while the mutant will raise
    a ValueError due to incompatible regex flags.
    """
    # A multi-line HTML string to match
    html_input = "<div>\n    <p>Hello, World!</p>\n</div>"
    output = HTML_RE.match(html_input)
    assert output is not None, "Expected to match, but didn't."