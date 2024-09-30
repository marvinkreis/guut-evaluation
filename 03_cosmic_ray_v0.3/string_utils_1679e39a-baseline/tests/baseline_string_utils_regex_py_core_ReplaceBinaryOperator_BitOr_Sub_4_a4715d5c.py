from string_utils._regex import HTML_RE

def test__HTML_RE():
    """
    Test if the regular expression HTML_RE correctly matches a simple HTML tag. The input represents a basic HTML
    paragraph tag. This test will pass for the correct code but fail for the mutant because the mutant incorrectly modifies
    the flags by subtracting re.DOTALL, causing it to miss multi-line HTML elements.
    """
    input_string = "<p>This is a paragraph.</p>"
    output = HTML_RE.match(input_string)
    assert output is not None