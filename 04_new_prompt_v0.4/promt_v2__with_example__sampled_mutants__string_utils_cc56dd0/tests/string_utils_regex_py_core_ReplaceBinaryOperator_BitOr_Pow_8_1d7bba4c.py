from string_utils._regex import PRETTIFY_RE

def test_prettify_left_space_mutant_killing():
    """
    Test the LEFT_SPACE regex from PRETTIFY_RE with a multiline string.
    The baseline will successfully match leading spaces before parentheses,
    while the mutant raises an OverflowError due to incorrect compilation.
    """
    test_string = "Here is a sentence with spaces before (parentheses).\nAnd another line with  (spaces)."
    matches = PRETTIFY_RE['LEFT_SPACE'].findall(test_string)
    assert matches != [], "Expected matches, got no matches."