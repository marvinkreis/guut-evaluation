from string_utils._regex import PRETTIFY_RE

def test_PRETTIFY_RE():
    # String with consecutive spaces which should be caught by the regex
    test_string_with_duplicates = "This is  a test    string."
    
    # Check matches in the original implementation
    matches_original = PRETTIFY_RE['DUPLICATES'].findall(test_string_with_duplicates)
    
    # Expect to find duplicates
    assert len(matches_original) > 0, "Expected to find duplicate spaces in the original."

    # Create a specific case for the mutant
    mutant_test_string = "Test with    only spaces."

    # The original should still find this due to multiple spaces
    original_matches = PRETTIFY_RE['DUPLICATES'].findall(mutant_test_string)
    assert len(original_matches) > 0, "Expected to find duplicates in the original for spaces."

    # Now let's check behavior that should fail on the mutant due to mishandling
    only_single_space_string = "This string has single spaces only."

    # This should yield no matches in the mutant 
    mutant_matches = PRETTIFY_RE['DUPLICATES'].findall(only_single_space_string)
    assert len(mutant_matches) == 0, "Expected not to find duplicates in the mutant due to incorrect handling."

# Note: This final approach checks simple spaces and ensures only conditions observable by the incorrect code.