from string_utils._regex import HTML_TAG_ONLY_RE

def test_html_tag_only_mutant_killing():
    """
    Test that the HTML_TAG_ONLY_RE regex properly matches HTML tags.
    The baseline will return a match object, while the mutant will raise a TypeError
    due to incorrect usage of operators in the regex pattern.
    """
    html_string = "<div></div>"
    output = HTML_TAG_ONLY_RE.match(html_string)
    assert output is not None, "Expected a match object, but got None."