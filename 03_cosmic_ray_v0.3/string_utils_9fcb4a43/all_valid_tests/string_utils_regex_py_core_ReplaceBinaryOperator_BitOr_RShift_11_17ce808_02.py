from string_utils._regex import PRETTIFY_RE

def test__prettify():
    """
    This test checks that the regex correctly captures multiple consecutive spaces.
    An expected difference should arise from the mutant due to 
    the improper use of the '>>' operator instead of '|'.
    """
    # Test string with intentional complex spacing and punctuation
    test_string = "Text   \twith     irregular  spacing;   extra   spaces."
    correct_match = PRETTIFY_RE['DUPLICATES'].findall(test_string)

    # Assert that there should be matches for the duplicate spaces.
    assert len(correct_match) > 0, "Correct regex must find duplicate spaces."

# Invoke the test function to run the test
test__prettify()