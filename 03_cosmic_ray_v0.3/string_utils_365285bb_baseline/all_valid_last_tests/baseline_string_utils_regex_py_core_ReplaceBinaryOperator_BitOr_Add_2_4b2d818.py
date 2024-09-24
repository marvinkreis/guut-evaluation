from string_utils._regex import WORDS_COUNT_RE

def test__WORDS_COUNT_RE():
    # A test phrase with clear word boundaries and punctuation
    test_string = "Cats and dogs are friends."

    # Get all matches
    matches = WORDS_COUNT_RE.findall(test_string)

    # Expected words: Cats, and, dogs, are, friends
    expected_words_count = 5
    actual_count = len(matches)

    # Assert the expected count of words
    assert actual_count == expected_words_count, f"Expected {expected_words_count} matches, but got {actual_count}."

    # Normalize matches to remove trailing spaces and punctuation
    normalized_matches = [word.strip(' .') for word in matches]

    # Expected words in correct form after normalization
    expected_words = ['Cats', 'and', 'dogs', 'are', 'friends']

    # Assert normalized matches against expected words
    assert normalized_matches == expected_words, f"Expected matches: {expected_words}, but got {normalized_matches}."

# This test specifically checks for word captures while normalizing any trailing spaces or punctuation marks.