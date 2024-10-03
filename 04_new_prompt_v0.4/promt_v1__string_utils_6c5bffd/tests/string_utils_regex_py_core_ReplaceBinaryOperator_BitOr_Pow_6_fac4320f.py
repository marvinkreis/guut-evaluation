from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_re():
    """
    This test verifies the behavior of the HTML_TAG_ONLY_RE regex compiled in the baseline and mutant.
    The mutant includes a syntax error due to the use of '**' instead of '|', which leads to an OverflowError
    during the import phase. The baseline should pass the tests, while the mutant should raise an exception.
    """
    test_string_1 = "<div></div>"  # Valid HTML tag
    test_string_2 = "<invalid>"     # Should match as an invalid tag
    test_string_3 = "Not a tag"     # Should not match
    test_string_4 = "<!doctype html>"  # Should match
    
    for test_string in [test_string_1, test_string_2, test_string_3, test_string_4]:
        match = HTML_TAG_ONLY_RE.match(test_string)
        assert match is not None if test_string in [test_string_1, test_string_2, test_string_4] else match is None
        print(f"Testing: '{test_string}' => Match: {bool(match)}")
        
# Call the test function directly for execution.
test__html_tag_only_re()