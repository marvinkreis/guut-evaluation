from string_utils._regex import HTML_RE

def test__HTML_RE():
    """The mutant changes '|' to '*' in HTML_RE, which prevents proper matching of HTML strings."""
    test_input = "<html><head><title>Title</title></head><body>Content</body></html>"
    match = HTML_RE.search(test_input)
    assert match is not None, "HTML_RE must match valid HTML input"