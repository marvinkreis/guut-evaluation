from string_utils._regex import PRETTIFY_RE

def test_PRETTIFY_RE():
    # Original test string with multiple consecutive spaces for the correct behavior
    test_string = "This  is  a   test   string.   And  it  includes   multiple   spaces."

    # Use the original regex which should successfully find the duplicates
    original_matches = PRETTIFY_RE['DUPLICATES'].findall(test_string)
    
    # Should find matches for duplicates in the original input
    assert len(original_matches) > 0, "Original PRETTIFY_RE did not match duplicates correctly."

    # Prepare a string without extra spaces for the mutant variant
    # This input should be simplified or normalized to ensure efficiency in finding duplicates
    mutant_string = "This is a test string. And it includes multiple spaces."

    # The mutant regex should not find any duplicates in this input
    mutant_matches = PRETTIFY_RE['DUPLICATES'].findall(mutant_string)

    # Expect no matches for the mutant as it interprets it differently
    assert len(mutant_matches) == 0, "Mutant PRETTIFY_RE should not find any duplicates; zero matches expected."

# Execute the test function
test_PRETTIFY_RE()