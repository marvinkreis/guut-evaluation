from string_utils._regex import HTML_TAG_ONLY_RE

def test_html_tag_only_re():
    # Test case: Valid multi-line HTML
    valid_multiline_html = """<div>
<span>Hello, World!</span>
</div>"""

    # This should match correctly with the original regex implementation
    assert HTML_TAG_ONLY_RE.search(valid_multiline_html) is not None, "Correct multi-line HTML should match! (Correct Code)"

    # Test case: Plain text input (should NOT match)
    plain_text = "This is just plain text."
    assert HTML_TAG_ONLY_RE.search(plain_text) is None, "Should not match plain text! (Correct Code)"

    # Test case: Malformed HTML should still yield a match
    malformed_html = "<div>Some invalid content without a closing tag"
    assert HTML_TAG_ONLY_RE.search(malformed_html) is not None, "Should match even with malformed HTML! (Correct Code)"

    # Adding an additional check for invalid HTML structure
    invalid_html_structure = "<div><not_a_tag></div>"
    assert HTML_TAG_ONLY_RE.search(invalid_html_structure) is not None, "Should match with invalid tag structure! (Correct Code)"