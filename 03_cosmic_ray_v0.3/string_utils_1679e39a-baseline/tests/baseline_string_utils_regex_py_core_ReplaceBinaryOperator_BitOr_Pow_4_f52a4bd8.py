from string_utils._regex import HTML_RE

def test__HTML_RE():
    """
    Test HTML_RE regex pattern on a typical HTML string input. 
    The original code uses the logical OR operator to combine MULTILINE and DOTALL flags,
    whereas the mutant incorrectly uses the exponentiation operator, which will lead to a TypeError.
    This test checks for a valid HTML string that should be matched by the regex if implemented correctly.
    """
    html_string = "<div>Hello World</div>"
    match = HTML_RE.match(html_string)
    assert match is not None