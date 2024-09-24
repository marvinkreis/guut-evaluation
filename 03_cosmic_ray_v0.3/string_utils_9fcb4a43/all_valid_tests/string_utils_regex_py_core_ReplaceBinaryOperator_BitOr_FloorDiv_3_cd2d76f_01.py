from string_utils._regex import HTML_RE

def test__HTML_RE_mixed_case():
    """The change in how flags are combined in the mutant should lead to failure when matching mixed-case HTML tags."""
    mixed_case_html = " <HTML> <DIV>Content</DIV> <p>Text</p> </HTML> "
    output = HTML_RE.findall(mixed_case_html)
    assert len(output) > 0, "HTML_RE must detect mixed-case HTML tags"