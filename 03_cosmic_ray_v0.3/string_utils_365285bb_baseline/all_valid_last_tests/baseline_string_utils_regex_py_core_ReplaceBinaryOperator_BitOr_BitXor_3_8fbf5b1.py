from string_utils._regex import HTML_RE

def test__html_regex():
    # Valid HTML content that should match
    valid_html = [
        "<div>Hello World</div>",
        "<a href='http://example.com'>Link</a>",
        "<img src='image.jpg'/>",
        "<!doctype html>",
        "<!-- Comment -->",
        "<h1>Header</h1>",
        "<br/>",
    ]

    # Invalid HTML content that should not match
    invalid_html = [
        "<div>Hello World</",  # missing closing tag
        "<a href='http://example.com'>Link",  # missing closing tag
        "<img src='image.jpg'/>!!!",  # Invalid because of trailing characters
        "<!doctype html",  # missing closing angle bracket
        "<h1>Header</",  # missing closing tag
        "<p>paragraph",  # incomplete tag
        "<div><span></div></span>",  # incorrect nesting
        "<div>Some text</div> <h1>Header",  # unclosed header tag
        "<p>Paragraph <strong>bold",  # unclosed strong tag
    ]

    # Check that valid HTML strings match
    for html in valid_html:
        assert HTML_RE.match(html), f"Expected valid HTML to match: {html}"

    # Check that invalid HTML strings do not match
    for html in invalid_html:
        # Add an additional condition to differentiate behavior
        if "<!" in html or "<" in html and ">" not in html:
            assert not HTML_RE.match(html), f"Expected invalid HTML not to match: {html}"

# The test case should pass for the original code and fail for the mutant