from string_utils._regex import WORDS_COUNT_RE

def test__words_count_regex():
    # Test case with sample string that should match as it contains several words
    test_string = "Hello world! This is a test."
    expected_match_count = 6  # Expected words are: "Hello", "world", "This", "is", "a", "test"
    
    # Find all matches using the original regex
    matches = WORDS_COUNT_RE.findall(test_string)
    
    # Assert that the number of matches is as expected
    assert len(matches) == expected_match_count, f"Expected {expected_match_count} matches, but got {len(matches)}"