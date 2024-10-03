from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only():
    # A simple HTML string containing a tag
    html_string = "<div>Hello, World!</div>"
    
    # The pattern should match the opening and closing div tags
    assert HTML_TAG_ONLY_RE.search(html_string) is not None, "The HTML_TAG_ONLY_RE should match valid HTML tags."
    
    # An invalid string that should not match (not an HTML tag)
    invalid_string = "Just some text without any HTML tags."
    assert HTML_TAG_ONLY_RE.search(invalid_string) is None, "The HTML_TAG_ONLY_RE should not match non-HTML strings."

    # A more complex HTML string to further ensure matching
    complex_html_string = "<p>Some <strong>formatted</strong> text.</p>"
    assert HTML_TAG_ONLY_RE.search(complex_html_string) is not None, "The HTML_TAG_ONLY_RE should match complex HTML tags."

    print("All assertions passed.")

# When executed with the original code, this should pass, but fail with the mutant