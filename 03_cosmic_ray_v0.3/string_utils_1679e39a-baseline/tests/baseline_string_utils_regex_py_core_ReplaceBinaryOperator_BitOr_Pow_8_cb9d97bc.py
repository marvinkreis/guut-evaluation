from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    """
    Test whether the PRETTIFY_RE regex correctly identifies a pattern with multiple spaces before parentheses,
    which is expected to match in the baseline. The mutant changes how the regex pattern is compiled with respect
    to space handling, leading to a failure in this expected match.
    """
    test_string = "This is  a sample text   (with parentheses) that should match."
    output = PRETTIFY_RE['DUPLICATES'].search(test_string)
    assert output is not None  # This should pass in the baseline, as DUPLICATES should match the repeated spaces