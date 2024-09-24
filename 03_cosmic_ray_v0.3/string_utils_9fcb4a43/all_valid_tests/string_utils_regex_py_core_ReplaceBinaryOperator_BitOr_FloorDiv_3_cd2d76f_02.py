from string_utils._regex import HTML_RE

def test__HTML_RE_complex_mismatch():
    """The mutation should lead to failure when matching complex HTML with mismatched and mixed-case tags."""
    complex_nested_html = "<HTMl><DiV>Content</DiV><p>Text</p><Div></p></HTMl>"
    output = HTML_RE.findall(complex_nested_html)
    assert len(output) > 2, "HTML_RE must detect all mismatched and mixed-case HTML tags"