from string_utils.validation import is_isbn_10

def test__is_isbn_10_invalid_input():
    """Test with a completely invalid ISBN that is guaranteed to raise an exception in the mutant."""
    invalid_input = 'invalid-isbn'
    
    # The assertion expects that is_isbn_10 should raise an InvalidInputError for the invalid input
    try:
        output = is_isbn_10(invalid_input)
        assert output is False, "Invalid ISBN input should return False."
    except InvalidInputError:
        assert False, "InvalidInputError was raised, indicating an issue with exception handling in the mutant."