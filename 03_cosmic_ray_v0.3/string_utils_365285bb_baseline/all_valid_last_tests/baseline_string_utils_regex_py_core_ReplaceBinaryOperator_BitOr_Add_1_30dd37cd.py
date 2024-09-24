from string_utils._regex import WORDS_COUNT_RE

def test__words_count_re():
    # Input string with direct spaces and punctuation
    test_string = "This is an example, with... spaces, and other punctuation! 123"

    # Expected output should properly count the words while excluding numbers and punctuation
    expected_output_original = 10  # "This", "is", "an", "example", "with", "spaces", "and", "other", "punctuation"

    # Count words using the WORDS_COUNT_RE
    matches = WORDS_COUNT_RE.findall(test_string)
    actual_output = len(matches)

    # Assert the expected output matches the actual output
    assert actual_output == expected_output_original, f"Expected {expected_output_original}, got {actual_output}"

# This test should pass for the correct implementation and fail for the mutant.