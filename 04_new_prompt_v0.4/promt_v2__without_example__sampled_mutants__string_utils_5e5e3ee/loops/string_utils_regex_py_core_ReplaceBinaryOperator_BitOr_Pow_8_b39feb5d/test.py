from string_utils._regex import PRETTIFY_RE

def test__prettify_re_duplicates():
    """
    Test the 'DUPLICATES' pattern in PRETTIFY_RE to ensure it correctly
    identifies sequences of duplicated spaces in a string. The baseline
    should pass successfully while the mutant, due to an error with regex
    compilation, should fail. This test effectively kills the mutant.
    """
    test_string = "This   is a test.  \nAnother line without excess spaces."
    
    # Use the 'DUPLICATES' regex from the PRETTIFY_RE dictionary
    duplicates_pattern = PRETTIFY_RE['DUPLICATES']
    matches = duplicates_pattern.findall(test_string)
    
    # Expect matches for duplicated spaces
    assert matches == ['   ', '  \n'], f"Expected matches: ['   ', '  \\n'], but got: {matches}"