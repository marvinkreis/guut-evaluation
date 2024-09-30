from string_utils._regex import HTML_RE

def test__HTML_RE():
    """
    Test whether HTML_RE correctly matches a valid HTML document type declaration.
    The input '<!DOCTYPE html>' should match the regex as it represents a proper doctype.
    The mutant changes the way regular expression options are combined,
    which may prevent the regex from matching this input correctly, allowing us 
    to differentiate between baseline and mutant behavior.
    """
    input_string = '<!DOCTYPE html>'
    output = HTML_RE.match(input_string)
    assert output is not None