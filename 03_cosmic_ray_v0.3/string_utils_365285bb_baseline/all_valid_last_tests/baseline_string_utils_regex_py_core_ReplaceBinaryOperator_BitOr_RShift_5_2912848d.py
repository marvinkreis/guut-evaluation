from string_utils._regex import HTML_TAG_ONLY_RE

def test_HTML_TAG_ONLY_RE():
    # Test case 1: Valid HTML string
    valid_html = '<div>Hello, World!</div>'
    match_valid = HTML_TAG_ONLY_RE.search(valid_html)
    assert match_valid is not None, "Expected match for valid HTML, got None!"

    # Test case 2: Valid HTML string with self-closing tag
    self_closing_html = '<img src="image.jpg" />'
    match_self_closing = HTML_TAG_ONLY_RE.search(self_closing_html)
    assert match_self_closing is not None, "Expected match for valid self-closing HTML, got None!"

    # Test case 3: Invalid HTML string (without any tags)
    invalid_string = 'Just some plain text.'
    no_match_invalid = HTML_TAG_ONLY_RE.search(invalid_string)
    assert no_match_invalid is None, "Expected no match for plain text, found one!"

    # Test case 4: Edge case with invalid formatting (extra opening tag)
    malformed_html = '<div><span>Text</div>'
    match_malformed = HTML_TAG_ONLY_RE.search(malformed_html)
    assert match_malformed is not None, "Expected match for partially valid HTML, got None!"

    # Test case 5: Completely invalid HTML without tags
    completely_invalid = 'Hello, World!'
    no_match_complete_invalid = HTML_TAG_ONLY_RE.search(completely_invalid)
    assert no_match_complete_invalid is None, "Expected no match for text without any tags, found one!"

    print("All tests passed successfully!")

# Uncomment below to run the test function directly
# test_HTML_TAG_ONLY_RE()