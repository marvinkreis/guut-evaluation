from string_utils._regex import PRETTIFY_RE

def test__prettify_regex():
    test_string = "This is a test string with multiple spaces    and   some punctuation..."
    
    # This input should match the regex pattern defined in PRETTIFY_RE['DUPLICATES']
    # for multiple spaces, commas, etc.
    
    # Check 'DUPLICATES' which should match multiple spaces
    matches = PRETTIFY_RE['DUPLICATES'].findall(test_string)
    
    # We expect to find matches for duplicate spaces
    assert matches != []  # should find at least one match for duplicate spaces
    
    # Test a string that has more than two consecutive spaces
    test_string_with_problems = "This    is   a   test."
    
    # Run the regex on this problematic test string
    matches_with_problems = PRETTIFY_RE['DUPLICATES'].findall(test_string_with_problems)
    
    # We expect matches indicating problems with spaces
    assert matches_with_problems != []  # should find at least one match
    
    # Print output for debugging
    print("Matches found in test string:", matches)
    print("Matches found in problematic string:", matches_with_problems)

# You can call this test function directly to see results
test__prettify_regex()