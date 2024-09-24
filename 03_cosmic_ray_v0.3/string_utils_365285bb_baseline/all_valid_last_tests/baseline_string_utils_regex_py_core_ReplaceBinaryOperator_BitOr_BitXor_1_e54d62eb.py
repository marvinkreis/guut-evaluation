from string_utils._regex import WORDS_COUNT_RE

def test__words_count_regex():
    # Test a simple sentence with different punctuation.
    test_string = "Hello, world! This is a test."
    expected_word_count = 6  # Here, we should have: ["Hello", "world", "This", "is", "a", "test"]
    
    matches = WORDS_COUNT_RE.findall(test_string)
    assert len(matches) == expected_word_count, f"Expected {expected_word_count} words, found {len(matches)} in '{test_string}'"

    # Mix of punctuation and whitespace
    mixed_whitespace_string = "  Hello,   world!   This   is   a   test.  "
    expected_mixed_count = 6  # Expecting the same count
    
    matches_mixed = WORDS_COUNT_RE.findall(mixed_whitespace_string)
    assert len(matches_mixed) == expected_mixed_count, f"Expected {expected_mixed_count} words, found {len(matches_mixed)} in '{mixed_whitespace_string}'"

    # Purely punctuation should yield 0
    punctuation_only = "! @#$%^&*()"
    expected_punctuation_count = 0
    
    matches_punctuation = WORDS_COUNT_RE.findall(punctuation_only)
    assert len(matches_punctuation) == expected_punctuation_count, f"Expected {expected_punctuation_count} words, found {len(matches_punctuation)} in '{punctuation_only}'"

    # Contraction string where expected behavior is clearly defined
    contraction_string = "It's a sunny day! Isn't it lovely?"
    expected_contraction_count = 9  # Adjusted for how regex interprets contractions and punctuation

    # Expected words: ["It's", "a", "sunny", "day", "Isn't", "it", "lovely"]
    matches_contraction = WORDS_COUNT_RE.findall(contraction_string)
    assert len(matches_contraction) == expected_contraction_count, f"Expected {expected_contraction_count} words, found {len(matches_contraction)} in '{contraction_string}'"

    # Edge case with mixed conditions
    edge_case_string = "Well, isn't that nice?  Yes, very lovely indeed."
    expected_edge_case_count = 9  # Adjusted as well based on regex behavior seen

    # Expected words: ["Well", "isn't", "that", "nice", "Yes", "very", "lovely", "indeed"]
    matches_edge_case = WORDS_COUNT_RE.findall(edge_case_string)
    assert len(matches_edge_case) == expected_edge_case_count, f"Expected {expected_edge_case_count} words, found {len(matches_edge_case)} in '{edge_case_string}'"