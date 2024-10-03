from string_utils._regex import HTML_RE

def test__html_re():
    """
    Test whether a well-formed HTML tag is recognized. The input string contains a simple HTML paragraph tag.
    The original regex matches this correctly, but the mutant incorrectly alters the re.DOTALL flag using a division,
    which would lead to a RegexError. Thus, this test will pass for the correct code and fail for the mutant.
    """
    output = HTML_RE.match('<p>This is a paragraph.</p>')
    assert output is not None