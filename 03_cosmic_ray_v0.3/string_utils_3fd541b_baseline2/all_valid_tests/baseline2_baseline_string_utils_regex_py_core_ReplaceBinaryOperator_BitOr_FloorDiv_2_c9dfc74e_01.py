from string_utils._regex import WORDS_COUNT_RE

def test__words_count_re():
    # This test checks if the regex can compile correctly and matches a simple sentence.
    
    # Original input for testing
    test_string = "Hello, world!"
    
    # Check if the regex compiles and matches
    try:
        # The regex should match and find one word
        matches = WORDS_COUNT_RE.findall(test_string)
        
        # Assert that we found two words: "Hello" and "world"
        assert len(matches) == 2, f"Expected 2 words, but found {len(matches)}: {matches}"
    except re.error as e:
        # If there's a compilation error, the mutant is detected
        assert False, f"Regex compilation failed: {str(e)}"