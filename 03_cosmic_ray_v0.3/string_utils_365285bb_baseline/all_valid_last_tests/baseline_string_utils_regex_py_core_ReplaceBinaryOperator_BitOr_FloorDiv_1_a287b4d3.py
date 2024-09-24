from string_utils._regex import WORDS_COUNT_RE

def test__words_count_regex():
    # A simple sample text
    sample_text = "Word1, Word2. Word3! Count: words here."  # Clear boundaries and various punctuation

    # Capture matches using the provided regex
    matches = WORDS_COUNT_RE.findall(sample_text)

    # Expected words from the sample text:
    # ["Word1", "Word2", "Word3", "Count", "words", "here"] - counted manually based on clarity
    expected_word_count = 6

    # Assert that the length of the matched words equals the expected word count
    assert len(matches) == expected_word_count, f"Expected {expected_word_count} words, but got {len(matches)}."

# This function should pass when regex operates correctly and fail with the mutant causing mismatches.