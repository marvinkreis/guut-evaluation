from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_re():
    # This test input includes HTML tags which should be successfully matched.
    test_input = "<div>Hello</div>"
    match = HTML_TAG_ONLY_RE.search(test_input)
    assert match is not None, "The regex should match a valid HTML tag."

    # Test input that contains no valid HTML tags.
    test_input_invalid = "This is not an HTML tag."
    no_match = HTML_TAG_ONLY_RE.search(test_input_invalid)
    assert no_match is None, "The regex should not match a non-HTML string."