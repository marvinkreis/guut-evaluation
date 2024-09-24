from string_utils._regex import PRETTIFY_RE

def test__prettify():
    """
    This test checks that the regex captures cases where excessive spaces 
    should be matched. The mutant's change from '|' to '>>' may cause 
    it to miss matches entirely.
    """

    # Test string concatenated with intentional leading/trailing spaces and multiple spaces
    test_string = "     This should not match anything special.    "
    correct_match = PRETTIFY_RE['DUPLICATES'].findall(test_string)

    # For leading and trailing spaces, we expect at least one match
    assert len(correct_match) > 0, "Correct regex must find matches for leading/trailing spaces."

# Invoke the test function
test__prettify()