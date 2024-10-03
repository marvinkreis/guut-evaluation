from string_utils._regex import HTML_RE

def test__HTML_RE():
    """
    Test the HTML_RE to check if it compiles and matches a simple HTML string.
    The test is expected to pass on the baseline but will fail on the mutant,
    since the mutant introduces a syntax error due to an incorrect operator.
    """
    test_string = "<html><body><h1>Test</h1></body></html>"
    output = HTML_RE.match(test_string)
    assert output is not None