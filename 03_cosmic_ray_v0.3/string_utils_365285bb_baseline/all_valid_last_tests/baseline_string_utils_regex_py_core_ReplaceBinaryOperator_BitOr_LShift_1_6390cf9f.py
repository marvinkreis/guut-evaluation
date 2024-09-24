from string_utils._regex import WORDS_COUNT_RE

def test_words_count():
    # Test input with mixed punctuation and spaces.
    test_string = "words - are not the same as: well; maybe they are."
    expected_word_count = 10  # words: "words", "are", "not", "the", "same", "as", "well", "maybe", "they", "are"
    
    # Find all words using the WORDS_COUNT_RE regex.
    words = WORDS_COUNT_RE.findall(test_string)
    
    # Validate that the count of words found matches the expected count.
    assert len(words) == expected_word_count, f"Expected {expected_word_count} words, but got {len(words)}."