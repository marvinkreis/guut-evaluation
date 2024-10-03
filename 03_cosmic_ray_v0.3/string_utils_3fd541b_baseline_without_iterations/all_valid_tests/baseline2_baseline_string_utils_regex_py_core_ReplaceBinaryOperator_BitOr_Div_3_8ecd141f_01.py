from string_utils._regex import HTML_RE

def test__html_regex():
    # Test with a valid HTML string, which should match the HTML_RE
    valid_html = '<div>Hello, World!</div>'
    assert HTML_RE.match(valid_html) is not None, "The valid HTML was not matched."

    # Test with a non-HTML string, which should not match the HTML_RE
    invalid_html = 'Just some text without tags.'
    assert HTML_RE.match(invalid_html) is None, "The invalid string was matched as HTML."

    # Additional HTML cases to ensure comprehensive coverage
    self_closing_tag = '<br />'
    assert HTML_RE.match(self_closing_tag) is not None, "Self-closing tag was not matched."

    comment = '<!-- This is a comment -->'
    assert HTML_RE.match(comment) is not None, "HTML comment was not matched."

    doctype = '<!doctype html>'
    assert HTML_RE.match(doctype) is not None, "Doctype declaration was not matched."

    # Edge cases
    empty_string = ''
    assert HTML_RE.match(empty_string) is None, "Empty string was matched as HTML."

    multiple_tags = '<div><span>Test</span></div>'
    assert HTML_RE.match(multiple_tags) is not None, "Multiple nested tags were not matched."

    print("All tests passed.")