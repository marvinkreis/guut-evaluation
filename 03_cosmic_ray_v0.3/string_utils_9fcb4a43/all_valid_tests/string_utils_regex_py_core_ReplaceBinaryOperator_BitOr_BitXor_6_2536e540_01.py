from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_regex():
    """Changing 're.DOTALL' in HTML_TAG_ONLY_RE for XOR might lead to missed matches."""
    
    # Edge test cases with malformed HTML
    malformed_html = """<html>
    <head>
    <title>Test<title>
    <body><h1>Header<h1></body>
    </html>"""
    
    output = HTML_TAG_ONLY_RE.findall(malformed_html)
    
    # Ensure it catches leading HTML tags at the start
    assert len(output) > 0, "HTML_TAG_ONLY_RE must capture HTML tags!"