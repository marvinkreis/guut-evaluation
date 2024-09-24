from string_utils._regex import WORDS_COUNT_RE
import re

def test_words_count_re():
    # New test string with mixed cases and additional complexities
    test_string = "Hello!@#   world!!  This is a TEST; should it count correctly? No, really! Count it! 😃"
    
    # Correct regex for matching valid words
    correct_words_re = r'\b\w+\b'  # This regex captures words, ignoring punctuation and spaces
    correct_matches = re.findall(correct_words_re, test_string)

    # Print the correct matches for debugging purposes
    print("Correct matches:", correct_matches)

    expected_count = 14  # The expected word count based on the input string

    # Assert for the correct number of words matched
    assert len(correct_matches) == expected_count  # Ensure expected count is correct

    # Define the mutant regex with faulty implementation
    mutant_words_count_re = re.compile(r'\W*[^\W_]+\W*', re.IGNORECASE | re.MULTILINE)  # Faulty usage of regex rules

    # Capturing the mutant matches
    mutant_matches = re.findall(mutant_words_count_re, test_string)

    # Print the raw output of the mutant matches
    print("Mutant matches (raw):", mutant_matches)

    # Clean the mutant matches
    cleaned_mutant_matches = [m.strip(' ,.!?@;') for m in mutant_matches]

    # Print cleaned mutant matches for comparison
    print("Cleaned Mutant matches:", cleaned_mutant_matches)

    # Check if the cleaned matches differ in count or content
    assert len(cleaned_mutant_matches) != len(correct_matches) or sorted(cleaned_mutant_matches) != sorted(correct_matches)

# Run the test function
test_words_count_re()