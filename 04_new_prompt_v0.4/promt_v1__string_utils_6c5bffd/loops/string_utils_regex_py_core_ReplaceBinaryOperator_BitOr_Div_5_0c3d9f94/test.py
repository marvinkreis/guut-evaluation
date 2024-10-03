from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_re():
    """
    Test the HTML_TAG_ONLY_RE regular expression to check its functionality.
    The test verifies that the regex correctly matches a sample HTML tag.
    The mutant introduces a syntax error due to incorrect operator usage,
    leading to a TypeError when importing.
    """
    try:
        sample_input = "<html></html>"
        match = HTML_TAG_ONLY_RE.match(sample_input)
        assert match is not None, "Should have matched the HTML tag"
    except Exception as e:
        assert isinstance(e, TypeError), f"Unexpected exception type: {type(e)}"