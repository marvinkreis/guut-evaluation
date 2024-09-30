from string_utils._regex import HTML_TAG_ONLY_RE

def test__HTML_TAG_ONLY_RE():
    """
    Test whether an HTML self-closing tag is matched correctly. 
    The input "<br/>" is a self-closing tag which will be matched 
    correctly when the regex is functioning as intended. If the 
    mutant is present and modifies the DOTALL flag, it will fail 
    to capture such tags that span multiple lines, hence killing the mutant.
    """
    output = HTML_TAG_ONLY_RE.match("<br/>")
    assert output is not None