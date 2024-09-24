from string_utils._regex import PRETTIFY_RE 

def test__PRETTIFY_RE():
    """Testing the impact of changing '|' to '>>' in PRETTIFY_RE."""
    
    # Case 1: Valid scenario with recognized duplicates using spaces
    test_string = "This is a string with multiple    spaces and formatting..."
    correct_matches = PRETTIFY_RE['DUPLICATES'].findall(test_string)
    
    assert len(correct_matches) > 0, "Correct PRETTIFY_RE should match duplicates (multiple spaces)."
    
    # Case 2: A scenario guaranteed to yield no matches due to not having duplicates
    invalid_string = "Just a simple test."
    invalid_matches = PRETTIFY_RE['DUPLICATES'].findall(invalid_string)
    
    assert len(invalid_matches) == 0, "Should find no matches in invalid scenario."

    # Introducing a case that heavily relies on multiple line breaks which the mutant might mishandle.
    complex_test_string = "Leading spaces     \n" \
                          "This line has   \n multiple duplicates   !\n  Multiple lines.\n\n" \
                          "Should it work?  \n\n  Or fail?    ...  "
    
    complex_matches = PRETTIFY_RE['DUPLICATES'].findall(complex_test_string)
    
    assert len(complex_matches) > 0, "Correct PRETTIFY_RE should match complex spaces and line breaks."
    
    # Verify the mutant reasoning by expecting the regex matches to fail (indicatively).
    assert len(complex_matches) > 1, "Expecting more matches in a complex multiline scenario."