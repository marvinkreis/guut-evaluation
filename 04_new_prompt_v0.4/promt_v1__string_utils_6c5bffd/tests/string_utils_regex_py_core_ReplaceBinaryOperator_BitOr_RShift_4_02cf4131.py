from string_utils._regex import HTML_RE

def test__html_regex_multiline_killer():
    """
    Test whether the HTML_RE regex properly matches multi-line HTML strings.
    The input is a multi-line HTML string which should pass for the baseline 
    but fail for the mutant due to improper handling of the DOTALL flag.
    """
    html_string = "<div>\nHello, World!</div>"
    match = HTML_RE.match(html_string)

    # Assert that the match should be successful and contain the whole string
    assert match is not None
    assert match.group(0) == "<div>\nHello, World!</div>"  # This assertion should fail on the mutant