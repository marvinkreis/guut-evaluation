from string_utils._regex import HTML_RE

def test__HTML_RE():
    """
    Test whether the HTML_RE can match an HTML tag with mixed case 
    and line breaks. The input '<DIV>\nContent\n</DIV>' is a valid 
    HTML tag which should match the regex string. The mutant's alteration 
    to incorrectly use '<<' will prevent it from correctly processing the 
    combination of flags leading it to fail to match valid HTML.
    """
    output = HTML_RE.match('<DIV>\nContent\n</DIV>')
    assert output is not None