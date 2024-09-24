from string_utils._regex import WORDS_COUNT_RE

def test__words_count_regex():
    # Create a string that tests various types of punctuation directly attached to words
    test_string = "Hello! What are you doing? Amazing!! こんにちは??! Isn't this beautiful? いい天気です."
    
    # Set the expected words without trailing punctuation
    expected_words = ["Hello", "What", "are", "you", "doing", "Amazing", "こんにちは", "Isn", "t", "this", "beautiful", "いい天気です"]
    
    # Find all matches using the regex
    matches = WORDS_COUNT_RE.findall(test_string)

    # Debug: Print found matches
    print("Matches found:", matches)

    # Normalize matches to ignore unwanted trailing punctuation
    matches_normalized = [match.strip(" !.,;:?'") for match in matches]
    
    # Debug: Print normalized matches
    print("Normalized matches:", matches_normalized)

    # Ensure matched results and expected outcomes correspond in count
    assert len(matches_normalized) == len(expected_words), f"Expected {len(expected_words)} words but got {len(matches_normalized)}: {matches_normalized}"
    
    # Compare the normalized results to the expected words
    assert sorted(matches_normalized) == sorted(expected_words), f"Matched words do not match expected: {matches_normalized}"

# Uncomment to execute the test manually
# test__words_count_regex()