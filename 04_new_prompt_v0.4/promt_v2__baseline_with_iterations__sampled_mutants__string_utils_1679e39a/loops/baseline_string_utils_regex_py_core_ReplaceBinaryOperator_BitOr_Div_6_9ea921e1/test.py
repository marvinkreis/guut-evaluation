from string_utils._regex import HTML_TAG_ONLY_RE

def test__HTML_TAG_ONLY_RE():
    """
    Test that the HTML_TAG_ONLY_RE correctly matches valid HTML tags. 
    The input '<div></div>' represents a valid HTML tag structure. 
    The mutant alters the regex by incorrectly replacing '|' with '/', 
    which will affect the matching behavior, causing this test to fail if the mutant is present.
    """
    output = HTML_TAG_ONLY_RE.match('<div></div>')
    assert output is not None