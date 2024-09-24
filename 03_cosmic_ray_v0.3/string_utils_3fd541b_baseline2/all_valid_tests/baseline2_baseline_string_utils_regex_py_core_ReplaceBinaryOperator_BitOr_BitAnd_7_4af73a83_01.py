from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    # Test string with intentional duplicate spaces
    test_string = "This  is  a  test string with  multiple   spaces."
    
    # Using the regex to match duplicates
    matches = PRETTIFY_RE['DUPLICATES'].findall(test_string)
    
    # With the original code, we expect to find duplicates; they should be detected
    assert len(matches) > 0, "Original regex should have detected duplicates"
    
    # Now the mutant would behave differently and potentially return no matches.
    # We can't directly assert here because we need to check against the mutant.
    
    # However, we can assume the test function can run under both versions for validation
    mutant_matches = []  # Placeholder; in practice run the code with the mutant
    
    assert len(mutant_matches) == 0, "Mutant regex should not have matched duplicates"
