from string_utils._regex import PRETTIFY_RE

def test__prettify():
    """
    The mutant changes the regex pattern from '|' to '>>', which can change the 
    behavior of matching duplicate spaces.
    """
    test_string = "This is   a test  string   with   extra   spaces."
    correct_match = PRETTIFY_RE['DUPLICATES'].findall(test_string)
    
    # Assert that there should be multiple matches for spaces.
    assert len(correct_match) > 1, "Correct regex must find multiple duplicate spaces."

# Invoke the test function
test__prettify()