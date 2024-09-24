from string_utils._regex import HTML_TAG_ONLY_RE

def test_html_tag_only_re_detection():
    # Test a well-formed HTML with one line
    html_string = "<div><span></span></div>"
    matches = HTML_TAG_ONLY_RE.findall(html_string)

    # Expecting: <div>, <span>, </span>, </div>
    assert len(matches) == 4, f"Expected 4 matches, but found {len(matches)}. Matches: {matches}"

    # Now testing with a variant which includes a newline in between to challenge flags
    html_string_multiline = "<div>\n<span>\n</span>\n</div>"
    multiline_matches = HTML_TAG_ONLY_RE.findall(html_string_multiline)

    # Expecting: 4 matches as it includes newline in the content
    assert len(multiline_matches) == 4, f"Expected 4 matches for multi-line, but found {len(multiline_matches)}. Matches: {multiline_matches}"

    # Test with no HTML tags at all
    no_tags_string = "This string contains no HTML."
    no_tags_matches = HTML_TAG_ONLY_RE.findall(no_tags_string)

    # Expecting no matches
    assert len(no_tags_matches) == 0, f"Expected 0 matches for no tags, but found {len(no_tags_matches)}. Matches: {no_tags_matches}"

# Run the test
test_html_tag_only_re_detection()