from string_utils.validation import words_count

def test__words_count():
    """The mutant incorrectly raises an InvalidInputError for valid string inputs. This test will detect it."""
    valid_input = "hello world"
    output = words_count(valid_input)
    assert output == 2, "words_count must return the correct word count for valid strings."