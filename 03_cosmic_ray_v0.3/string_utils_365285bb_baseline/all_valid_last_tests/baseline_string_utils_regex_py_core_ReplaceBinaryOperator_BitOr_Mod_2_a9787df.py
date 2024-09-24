from string_utils._regex import WORDS_COUNT_RE

def test__words_count_regex():
    # Simple test string with clear word boundaries
    test_string = "Birds and cats are pets."

    # Expected words: ["Birds", "and", "cats", "are", "pets"]
    expected_word_count = 5  # There should be five words

    # Use the regex to find all word occurrences
    found_words = WORDS_COUNT_RE.findall(test_string)

    # Print the found words for debugging
    print("Found words before cleanup:", found_words)

    # Clean up found words by stripping trailing whitespace and filtering out any that aren't purely alphabetic
    cleaned_words = [word.strip(". ") for word in found_words]  # Remove trailing punctuation and spaces

    # Print cleaned words for debugging
    print("Cleaned words:", cleaned_words)

    # Assert that the number of cleaned words matches the expected count
    assert len(cleaned_words) == expected_word_count, f"Expected {expected_word_count} words but got {len(cleaned_words)}"

    # Validate that all cleaned words consist of valid alphabetic characters
    assert all(word.isalpha() for word in cleaned_words), "All found words should consist of alphabetical characters only."