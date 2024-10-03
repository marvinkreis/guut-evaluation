from string_utils._regex import HTML_RE

def test__html_regex_complexity():
    """
    Test whether the HTML_RE regex can correctly match complex multiline HTML content, 
    including nested elements and comments. The baseline should match the entire string,
    while the mutant will only match the opening <div> tag, demonstrating the loss of functionality.
    """
    test_string = """<div>
    <!-- A comment -->
    <span>Hello</span>
    </div>
    <p>Paragraph.</p>"""

    match = HTML_RE.match(test_string)
    assert match is not None, "Expected a match with the regex pattern."
    assert match.span() == (0, 51), f"Expected match span of (0, 51), got {match.span()}"