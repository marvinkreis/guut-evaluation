from string_utils._regex import PRETTIFY_RE

def test_pretify_regex_mutant_killing():
    """
    Test the PRETTIFY_RE dictionary patterns. Specifically, we will check the 
    'DUPLICATES' pattern for multiple spaces and the 'RIGHT_SPACE' pattern 
    for spaces adjacent to punctuation. The mutant will cause a ValueError,
    while the baseline will correctly find matches.
    """
    # Check for duplicate spaces
    test_input_duplicates = "This is a test   and   should not have   duplicates."
    matches_duplicates = PRETTIFY_RE['DUPLICATES'].findall(test_input_duplicates)
    assert matches_duplicates != [], "Expected at least one match for duplicates but got none."

    # Check for right space issues
    test_input_right_space = "This is a test , that has no right space ."
    matches_right_space = PRETTIFY_RE['RIGHT_SPACE'].findall(test_input_right_space)
    assert matches_right_space != [], "Expected at least one match for right space but got none."