from string_utils._regex import HTML_RE

def test__html_regex_differentiator():
    """
    Test to check if the regex for HTML matches a complex HTML string correctly.
    The baseline should return a match object for the entire snippet,
    while the mutant should fail to match it correctly.
    This confirms that the change introduced in the mutant affects regex behavior.
    """
    html_string = '''<div>
        <p>Hello World</p>
        <hr />
      </div>'''
    match = HTML_RE.match(html_string)
    assert match is not None and match.span() == (0, 32)  # Expecting full match