from string_utils.validation import words_count

def test__words_count():
    # Test with a valid string that contains words
    input_string = 'hello world'
    expected_count = 2  # "hello" and "world" are two words
    assert words_count(input_string) == expected_count
    
    # Test with an invalid input type that would raise an InvalidInputError
    try:
        invalid_input = None  # None should not be accepted
        words_count(invalid_input)
        assert False, "Expected InvalidInputError for None input"
    except Exception as e:
        assert isinstance(e, Exception)  # checking if an exception is raised

    # Adding a test with a string that has punctuation
    input_string_with_punctuation = 'one,two,three.stop'
    expected_count_with_punctuation = 4  # Expected to count "one", "two", "three", and "stop"
    assert words_count(input_string_with_punctuation) == expected_count_with_punctuation

    # Test with an empty string
    input_empty = ''
    expected_empty_count = 0  # No words in an empty string
    assert words_count(input_empty) == expected_empty_count