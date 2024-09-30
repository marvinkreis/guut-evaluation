from string_utils._regex import HTML_RE

def test__HTML_RE():
    """
    Test whether the HTML_RE correctly matches a valid HTML tag. The input represents a simple HTML tag: 
    <div></div>. The expected behavior is that it matches the entire string. The mutant replaces the 
    '|' operator with a '/', which should cause this test to fail because the regex would not function correctly anymore.
    """
    output = HTML_RE.fullmatch("<div></div>")
    assert output is not None