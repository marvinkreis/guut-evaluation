from string_utils._regex import PRETTIFY_RE

def test__prettify():
    """
    This test checks multiple attributes of the regex, specifically that it correctly captures
    duplicate spaces across different contexts, especially when they are surrounded by punctuation.
    The mutant's logical change should cause it to act differently.
    """

    # Test input string with mixed spaces and punctuation
    test_string = "Hello   world!    Test   string;    with   extra   spaces?"

    # Expected behavior is to find duplicates of spaces here.
    correct_match = PRETTIFY_RE['DUPLICATES'].findall(test_string)

    print(f"Matches from correct regex: {correct_match}")
    
    # Assert that we expect at least one match for duplicates
    assert len(correct_match) > 0, "Correct regex must find matching duplicates."

# Run the test
test__prettify()