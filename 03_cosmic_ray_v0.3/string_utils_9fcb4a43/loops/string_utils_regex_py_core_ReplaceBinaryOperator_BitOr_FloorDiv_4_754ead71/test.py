from string_utils._regex import HTML_RE

def test__HTML_RE_nested_multiline():
    """The mutant's changes to HTML_RE will lead it to incorrectly match nested HTML structures."""
    nested_multiline_test_string = '<div>\n<span>Hello</span>\nWorld!\n</div>'
    
    # Expecting successful match with correct HTML_RE
    output = HTML_RE.findall(nested_multiline_test_string)
    
    # Verify that output should contain the full structure as a single match
    assert len(output) == 1, "HTML_RE must match the entire nested structure as one match"
    assert output[0][0] == '<div>\n<span>Hello</span>', "HTML_RE must capture the complete nested HTML content"