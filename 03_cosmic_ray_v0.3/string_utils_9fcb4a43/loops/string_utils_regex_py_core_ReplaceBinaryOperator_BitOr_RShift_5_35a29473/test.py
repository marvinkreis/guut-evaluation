from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_re():
    """The mutant implementation fails to match HTML tags as expected."""
    html_input = "<HTML><BODY>Some text.</BODY></HTML>"
    
    # This should return matches for any valid HTML tags
    correct_matches = HTML_TAG_ONLY_RE.findall(html_input)
    assert len(correct_matches) > 0, "HTML_TAG_ONLY_RE must match HTML tags"