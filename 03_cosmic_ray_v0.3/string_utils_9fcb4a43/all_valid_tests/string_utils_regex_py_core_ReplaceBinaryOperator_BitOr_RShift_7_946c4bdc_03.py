def test__regex_behavior():
    """Ensure the regex behaviors for spaces and uppercase letters behave differently in the mutant."""
    from string_utils._regex import PRETTIFY_RE
    
    sample_text = "Extra     spaces   are   not   nice."
    
    try:
        # Correct code should find matches in this text
        match_duplicates = PRETTIFY_RE['DUPLICATES'].search(sample_text)
        assert match_duplicates is not None, "Expected duplicate spaces to be matched."
        
        match_upper = PRETTIFY_RE['UPPERCASE_FIRST_LETTER'].search(sample_text)
        assert match_upper is not None, "Expected an uppercase letter to be matched."
        
    except Exception as e:
        print(f"An error occurred while validating regex behavior: {e}")

# Run the test
test__regex_behavior()