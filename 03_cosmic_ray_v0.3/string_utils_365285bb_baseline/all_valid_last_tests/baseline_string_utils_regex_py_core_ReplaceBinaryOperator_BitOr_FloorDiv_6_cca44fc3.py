from string_utils._regex import HTML_TAG_ONLY_RE

def test_HTML_TAG_ONLY_RE():
    # Valid multi-line HTML that should match
    valid_html_multiline = "<html>\n<body>\nSome content\n</body>\n</html>"
    assert HTML_TAG_ONLY_RE.search(valid_html_multiline) is not None, "Valid multi-line HTML should be detected."

    # Valid HTML with nested tags that should also match
    valid_nested_html = "<div><p>Some</p> <p>Content</p></div>"
    assert HTML_TAG_ONLY_RE.search(valid_nested_html) is not None, "Valid nested HTML should be detected."

    # Invalid input that should not match (non-HTML)
    invalid_html = "This is just text."
    assert HTML_TAG_ONLY_RE.search(invalid_html) is None, "Non-HTML content should not match."

    # Test a well-formed HTML but slightly malformed (still should match)
    malformed_html = "<div>Content without a closing tag"
    assert HTML_TAG_ONLY_RE.search(malformed_html) is not None, "Malformed but valid structure should still match."

    # Test for an HTML tag that is incorrectly written
    incorrect_html = "<html><body>Text without </body>"
    assert HTML_TAG_ONLY_RE.search(incorrect_html) is not None, "Should match incorrect but structured HTML."

    # More complex multiline HTML
    complex_html = "<html>\n<head><title>Test</title></head>\n<body>\n<div>Content</div>\n</body></html>"
    assert HTML_TAG_ONLY_RE.search(complex_html) is not None, "Complex multi-line HTML should be detected."

    print("All tests passed.")

# Call the test function to execute the tests
test_HTML_TAG_ONLY_RE()