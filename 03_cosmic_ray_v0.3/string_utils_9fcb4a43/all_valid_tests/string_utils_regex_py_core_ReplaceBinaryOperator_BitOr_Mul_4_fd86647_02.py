from string_utils._regex import HTML_RE

def test__HTML_RE():
    """The mutant changes '|' to '*' in HTML_RE, which prevents proper matching of HTML strings."""
    test_inputs = [
        "<html><head><title>Title</title></head><body>Content</body></html>",
        "<div class='test'>Test Content</div>",
        "<a href='https://example.com'>Example</a>",
        "<img src='image.png' alt='Image'>",
        "<span>Some text<p>Another paragraph</p></span>"
    ]
    
    for input_html in test_inputs:
        match = HTML_RE.search(input_html)
        assert match is not None, f"HTML_RE must match valid HTML input for: {input_html}"