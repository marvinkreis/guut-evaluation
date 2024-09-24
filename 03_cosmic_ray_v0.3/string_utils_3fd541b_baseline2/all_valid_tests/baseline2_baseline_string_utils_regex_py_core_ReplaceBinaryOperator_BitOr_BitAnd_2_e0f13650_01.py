from string_utils._regex import WORDS_COUNT_RE

def test__words_count_re():
    # Test case for the original WORDS_COUNT_RE regex
    valid_string = "Hello, world! This is a test."
    invalid_string = "!!!####@@@"
    
    # This should match as it has useful words
    assert WORDS_COUNT_RE.search(valid_string) is not None, "valid_string should match WORDS_COUNT_RE"
    
    # This should not match as it only contains punctuation
    assert WORDS_COUNT_RE.search(invalid_string) is None, "invalid_string should not match WORDS_COUNT_RE"