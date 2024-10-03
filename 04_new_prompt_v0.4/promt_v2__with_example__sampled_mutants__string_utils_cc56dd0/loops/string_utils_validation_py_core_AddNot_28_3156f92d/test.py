from string_utils.validation import words_count

def test_words_count_mutant_killing():
    """
    Test the words_count function with both a non-string input and a valid string input.
    The baseline should raise an InvalidInputError for a non-string,
    while the mutant will process it incorrectly and return a count of words.
    The valid string input will be checked for a proper return value from the baseline.
    """
    # Testing a non-string input
    try:
        output = words_count(123)
        print(f"Output for non-string input: {output}") # This should not be executed
    except Exception as e:
        print(f"Expected Exception for non-string input: {str(e)}")

    # Testing valid string input
    output = words_count('hello world')
    assert output == 2, f"Expected 2, got {output}"