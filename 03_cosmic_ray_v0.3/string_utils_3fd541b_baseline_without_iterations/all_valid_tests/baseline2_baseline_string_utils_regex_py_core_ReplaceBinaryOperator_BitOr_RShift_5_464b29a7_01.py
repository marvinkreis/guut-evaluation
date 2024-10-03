from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_re():
    # Valid HTML tag (self-closing)
    valid_tag = "<img src='image.png'/>"
    assert HTML_TAG_ONLY_RE.match(valid_tag) is not None, "Expected a match for valid HTML tag"

    # Valid closing HTML tag
    valid_closing_tag = "</div>"
    assert HTML_TAG_ONLY_RE.match(valid_closing_tag) is not None, "Expected a match for valid closing HTML tag"

    # Invalid HTML tag (missing closing bracket)
    invalid_tag = "<img src='image.png'"
    assert HTML_TAG_ONLY_RE.match(invalid_tag) is None, "Expected no match for invalid HTML tag"

    # Another valid HTML tag
    another_valid_tag = "<a href='http://example.com'></a>"
    assert HTML_TAG_ONLY_RE.match(another_valid_tag) is not None, "Expected a match for another valid HTML tag"