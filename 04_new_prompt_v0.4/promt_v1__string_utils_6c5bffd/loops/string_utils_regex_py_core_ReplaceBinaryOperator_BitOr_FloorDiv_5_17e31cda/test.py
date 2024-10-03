from string_utils._regex import HTML_TAG_ONLY_RE

def test__mixed_case_html_tags():
    """
    Test that the regex HTML_TAG_ONLY_RE correctly identifies mixed case HTML tags.
    The baseline should match the mixed case tag <Div>, while the mutant will fail to match 
    due to the change in the regex, making the behavior of the mutant inconsistent with 
    the intended case-insensitivity.
    """
    test_input = "<Div>example</Div>"
    match = HTML_TAG_ONLY_RE.match(test_input)

    print(f"match = {match}")
    assert match is not None