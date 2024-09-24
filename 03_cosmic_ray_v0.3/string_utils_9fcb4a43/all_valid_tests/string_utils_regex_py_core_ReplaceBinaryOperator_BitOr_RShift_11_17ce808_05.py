from string_utils._regex import PRETTIFY_RE

def test__prettify():
    """
    This test checks the ability of the regex to correctly find spaces in various contexts, 
    especially surrounding punctuation and other characters where spacing conditions may lead to 
    discrepancies due to the mutant's logical operator switch from '|' to '>>'.
    """

    # This string includes varied conditions of spacing around punctuation and text.
    test_string = "This is a test...          What?  Yes,    indeed!"
    
    # Find duplicate spaces using the correct implementation.
    correct_match = PRETTIFY_RE['DUPLICATES'].findall(test_string)

    # We expect matches specifically around double spacing and punctuation
    print(f"Matches from correct regex: {correct_match}")

    # Assert that we should have at least expected to find spaces
    assert len(correct_match) > 0, "Correct regex must find matching duplicates."

# Run the test
test__prettify()