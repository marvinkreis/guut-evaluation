from string_utils._regex import HTML_TAG_ONLY_RE

def test_html_tag_only_re_mutant_killing():
    """
    Test the HTML_TAG_ONLY_RE regex against an HTML string that spans multiple lines.
    The baseline should correctly find all HTML tags, while the mutant raises a ValueError
    due to incompatible flag settings.
    """
    html_string = "<div>\n    <p>Hello World!</p>\n</div>"
    try:
        matches = HTML_TAG_ONLY_RE.findall(html_string)
        assert matches != [], "Expected matches to be found, but got none."
    except ValueError as e:
        raise AssertionError("Expected to find matches, but raised ValueError: " + str(e))