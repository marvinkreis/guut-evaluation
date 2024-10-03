from string_utils._regex import PRETTIFY_RE

def test__prettify_re_multiline_case_test():
    """
    Verify that the PRETTIFY_RE correctly matches duplicates in a multi-line string.
    The input contains spaces that should trigger the 'DUPLICATES' match.
    The mutant is expected to fail on this input as it does not support multi-line matching.
    """
    input_string = "This  is a test.\n\nThis  should  match."
    match = PRETTIFY_RE['DUPLICATES'].search(input_string)  # Using search to find matches across lines
    assert match is not None, "The regex should match the input string with duplicated spaces across lines."