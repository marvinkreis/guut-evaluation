from string_utils.validation import words_count
from string_utils.errors import InvalidInputError

def test_words_count():
    # Test case where input is not a string
    try:
        words_count(None)  # should raise InvalidInputError
        assert False, "Expected InvalidInputError was not raised."
    except InvalidInputError:
        pass  # This is expected

    try:
        words_count(123)  # should raise InvalidInputError
        assert False, "Expected InvalidInputError was not raised."
    except InvalidInputError:
        pass  # This is expected

    # Test case with a valid string
    result = words_count("hello world")
    assert result == 2, f"Expected 2, got {result}"

    result = words_count("one,two,three.stop")
    assert result == 4, f"Expected 4, got {result}"