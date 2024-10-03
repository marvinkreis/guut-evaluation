from string_utils._regex import HTML_RE

def test__html_re():
    """
    Test whether a typical HTML tag with mixed case is matched correctly. 
    The input "<Div></DiV>" should match the HTML_RE regex in the baseline since it has re.IGNORECASE applied correctly.
    However, the mutant introduces a faulty bitwise shift operation which will cause this matching to fail for this particular case.
    """
    output = HTML_RE.match("<Div></DiV>")
    assert output is not None