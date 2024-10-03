from string_utils._regex import PRETTIFY_RE

def test__prettify_regex():
    # Test strings that shouldn't match based on prettify rules
    test_string_valid = "This is a valid sentence."
    test_string_invalid = "This  is  invalid  due  to  multiple   spaces."
    
    # We expect the valid string to match (not cause any replacements)
    assert PRETTIFY_RE['DUPLICATES'].search(test_string_valid) is None
    
    # We expect the invalid string to match (it contains duplicates)
    assert PRETTIFY_RE['DUPLICATES'].search(test_string_invalid) is not None
    
    # If we can reach this point, the original regex works correctly,
    # but if the mutant is in place, PRETTIFY_RE cannot be evaluated correctly.