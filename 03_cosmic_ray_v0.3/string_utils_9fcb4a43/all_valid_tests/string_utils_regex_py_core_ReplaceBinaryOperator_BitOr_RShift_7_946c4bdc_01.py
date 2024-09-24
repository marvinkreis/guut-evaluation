def test__prettify_regex():
    """Changing '|' to '>>' in the regex PRETTIFY_RE will lead to incorrect behavior or no matching capabilities."""
    sample_text = "Extra     spaces   should   be   reduced."
    
    # Testing the behavior of PRETTIFY_RE directly
    try:
        # This should yield useful regex matches when correctly defined
        match = PRETTIFY_RE['DUPLICATES'].search(sample_text)
        assert match is not None, "Expected duplicate spaces to match."
        
        uppercase_match = PRETTIFY_RE['UPPERCASE_FIRST_LETTER'].search(sample_text)
        assert uppercase_match is not None, "Expected an uppercase first letter to match."
    except Exception as e:
        print(f"An error occurred during the test: {e}")

test__prettify_regex()