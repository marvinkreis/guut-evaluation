from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_re():
    """
    Test if the HTML_TAG_ONLY_RE regex correctly matches a multi-line HTML tag.
    The input involves an HTML tag that spans multiple lines. The baseline should match it,
    while the mutant should raise a ValueError due to incompatible regex flags.
    """
    input_string = "<div>\n    <span>Hello</span>\n</div>"
    match = HTML_TAG_ONLY_RE.search(input_string)
    assert match is not None  # Baseline should match