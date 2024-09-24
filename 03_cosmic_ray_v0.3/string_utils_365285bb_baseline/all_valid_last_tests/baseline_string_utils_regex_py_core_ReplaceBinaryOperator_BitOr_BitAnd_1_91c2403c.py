from string_utils._regex import WORDS_COUNT_RE
import re

def test_words_count_re():
    # Input string containing various punctuation
    test_string = "Hello, I'm learning Python! This should count the words."

    # Use the regex to find matches
    matches = WORDS_COUNT_RE.findall(test_string)

    # Clean the matched words from punctuation
    cleaned_matches = [
        re.sub(r'[^\w\'-]', '', match).strip() for match in matches if match
    ]

    # Expected list of words we want to see including contraction variation
    expected_matches = ["Hello", "I'm", "learning", "Python", "This", "should", "count", "the", "words"]

    # Normalizing cleaned matches for checking against expected matches
    def normalize_word_list(words):
        normalized = []
        i = 0
        while i < len(words):
            word = words[i]
            # Check if we find "I'" and the next word is "m", to combine them as "I'm"
            if word == "I'" and (i + 1 < len(words) and words[i + 1] == 'm'):
                normalized.append("I'm")
                i += 2  # Skip the next 'm'
            else:
                normalized.append(word)
                i += 1
        return normalized

    normalized_matches = normalize_word_list(cleaned_matches)

    # Assert that all expected words are found in normalized matches
    for expected in expected_matches:
        assert expected in normalized_matches, \
            f"Expected word '{expected}' not found in matches: {normalized_matches}"

    # Additional check to ensure counts match
    assert len(normalized_matches) == len(expected_matches), \
        f"Expected {len(expected_matches)} matches, but got {len(normalized_matches)}"