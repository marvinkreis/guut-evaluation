from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    """Testing the impact of changing '|' to '>>' in PRETTIFY_RE."""
    
    # Case where duplicate spaces are well defined
    test_string_valid = "This    is a test   string with multiple spaces."
    correct_matches_valid = PRETTIFY_RE['DUPLICATES'].findall(test_string_valid)
    
    # Here we expect some matching duplicates
    assert len(correct_matches_valid) > 0, "Correct PRETTIFY_RE should match duplicates in valid case."
    
    # Case where no valid duplicates should be captured
    test_string_invalid = "No duplicates here!"
    incorrect_matches = PRETTIFY_RE['DUPLICATES'].findall(test_string_invalid)
    
    # Should find no matches for duplicates in a valid scenario without spaced repetitions.
    assert len(incorrect_matches) == 0, "No matches should be found in the invalid case."