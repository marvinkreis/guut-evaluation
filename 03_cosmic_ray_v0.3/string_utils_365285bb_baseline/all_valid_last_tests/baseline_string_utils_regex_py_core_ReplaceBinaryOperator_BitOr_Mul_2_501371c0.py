from string_utils._regex import WORDS_COUNT_RE

def test__words_count_regex():
    # Input string with diverse punctuation and space conditions
    test_string = "Hello, world! This is a test. Check regex working! YAY."

    # The number of unique words we expect to be found in the input string
    expected_match_count = 10  # Words are: "Hello", "world", "This", "is", "a", "test", "Check", "regex", "working", "YAY"

    # Performing the regex matching
    matches = WORDS_COUNT_RE.findall(test_string)

    # Check and assert the count of matched results
    assert len(matches) == expected_match_count, f"Expected {expected_match_count} matches, but got {len(matches)}"

    # The mutant should give either more or fewer than expected
    mutant_fail_counts = [expected_match_count - 1, expected_match_count + 1]  # One less or one more than expected

    # Ensure that the count is not consistent with possible mutant outcomes
    for incorrect_count in mutant_fail_counts:
        assert len(matches) != incorrect_count, f"Count should differ from mutant expectation, found {len(matches)} matches which should differ from {incorrect_count}."

    # Verify that at least one match is found which shows functionality
    assert len(matches) > 0, "Expected at least one match in the test string."