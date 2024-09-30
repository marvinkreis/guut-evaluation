from string_utils._regex import HTML_TAG_ONLY_RE

def test__HTML_TAG_ONLY_RE():
    """
    Test whether the regex for matching HTML tags only works correctly.
    The input contains a self-closing tag, which should match. The mutant change introduces an error 
    in the regex by replacing '|' with '**', causing it to fail on valid HTML syntax.
    """
    output = HTML_TAG_ONLY_RE.match("<img src='image.png'/>")
    assert output is not None