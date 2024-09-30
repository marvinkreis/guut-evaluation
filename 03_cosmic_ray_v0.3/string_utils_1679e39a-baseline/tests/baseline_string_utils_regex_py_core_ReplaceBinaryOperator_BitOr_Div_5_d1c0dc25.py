from string_utils._regex import HTML_TAG_ONLY_RE

def test__HTML_TAG_ONLY_RE():
    """
    Test whether a simple HTML tag is correctly matched. The input '<div>' should match the regex,
    while an invalid input without tags like 'just text' should not. The mutant alters the regex
    to use a division operator instead of the bitwise OR operator, which will cause the regex
    to not match correctly, thus this test will fail on the mutant.
    """
    valid_output = HTML_TAG_ONLY_RE.match('<div>') is not None
    invalid_output = HTML_TAG_ONLY_RE.match('just text') is None
    assert valid_output is True
    assert invalid_output is True