from string_utils._regex import WORDS_COUNT_RE

def test__words_count_regex():
    # Test input that should match the original regex behavior
    test_string = "Hello, how are you today?"
    
    # Originally, the regex should correctly identify words in the string
    match = WORDS_COUNT_RE.findall(test_string)
    
    # There are 5 words in the sentence, so we expect 5 matches
    assert len(match) == 5, f"Expected 5 words, found {len(match)}"

    # Now test for a case that would cause the mutant to fail
    # This input has punctuation but should be counted correctly
    test_string_with_punctuation = "Hi! This is an example."
    
    match_with_punctuation = WORDS_COUNT_RE.findall(test_string_with_punctuation)
    
    # Again, we expect 5 matches
    assert len(match_with_punctuation) == 5, f"Expected 5 words, found {len(match_with_punctuation)}"