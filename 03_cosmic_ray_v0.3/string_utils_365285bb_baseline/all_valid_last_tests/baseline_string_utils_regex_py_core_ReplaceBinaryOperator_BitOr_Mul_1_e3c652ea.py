from string_utils._regex import WORDS_COUNT_RE

def test_WORDS_COUNT_RE():
    # Test case 1: Basic sentence
    test_string_1 = "Hello, world! This is a test."
    expected_word_count_1 = 6  # Expected words: ["Hello", "world", "This", "is", "a", "test"]

    matches_1 = WORDS_COUNT_RE.findall(test_string_1)
    actual_word_count_1 = len(matches_1)

    # Assert for the first test
    assert actual_word_count_1 == expected_word_count_1, f"Expected {expected_word_count_1} but got {actual_word_count_1}"

    # Test case 2: Sentence with contractions
    test_string_2 = "Wow!!! Isn't it a test well?"
    expected_word_count_2 = 7  # Expected: ["Wow", "Isn't", "it", "a", "test", "well"]

    matches_2 = WORDS_COUNT_RE.findall(test_string_2)
    actual_word_count_2 = len(matches_2)

    # Assert for the second test
    assert actual_word_count_2 == expected_word_count_2, f"Expected {expected_word_count_2} but got {actual_word_count_2}"

    # Test case 3: Extra punctuation handling
    test_string_3 = "   ...   Extra   spaces   and   punctuation   !!!   "
    expected_word_count_3 = 4  # Expected: ["Extra", "spaces", "and", "punctuation"]

    matches_3 = WORDS_COUNT_RE.findall(test_string_3)
    actual_word_count_3 = len(matches_3)

    # Assert for the third test
    assert actual_word_count_3 == expected_word_count_3, f"Expected {expected_word_count_3} but got {actual_word_count_3}"

    # Test case 4: All punctuation
    test_string_4 = "!!!@@@###$$$%%%^^^&&&***((()))"
    expected_word_count_4 = 0  # No words should match

    matches_4 = WORDS_COUNT_RE.findall(test_string_4)
    actual_word_count_4 = len(matches_4)

    # Assert no words are found
    assert actual_word_count_4 == expected_word_count_4, f"Expected {expected_word_count_4} but got {actual_word_count_4}"

    # Test case 5: Edge case - only one 'word' composed entirely of non-space characters.
    test_string_5 = "onlynonspacedword"
    expected_word_count_5 = 1  # Should count the single word

    matches_5 = WORDS_COUNT_RE.findall(test_string_5)
    actual_word_count_5 = len(matches_5)

    # Assert for the fifth test
    assert actual_word_count_5 == expected_word_count_5, f"Expected {expected_word_count_5} but got {actual_word_count_5}"

# Execute the above function to perform testing