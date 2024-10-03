from string_utils._regex import HTML_TAG_ONLY_RE

def test__HTML_TAG_ONLY_RE():
    """
    This test checks whether the regex correctly matches a standard HTML tag.
    The input is a simple <div> tag. The output should be True if the regex
    is working correctly. The mutant modifies the regex by incorrect subtraction of re.MULTILINE,
    which would affect its matching capabilities for multi-line tags, causing this specific test to fail.
    """
    input_string = "<div>"
    output = HTML_TAG_ONLY_RE.match(input_string)
    assert output is not None