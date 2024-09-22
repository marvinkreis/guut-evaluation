from string_utils.manipulation import shuffle
from string_utils.errors import InvalidInputError

def test_shuffle():
    # Test with a simple valid string
    valid_string = "hello"
    
    # Call shuffle and check it does not throw an error
    try:
        result = shuffle(valid_string)
        
        # The output should not be the same as the input string
        assert result != valid_string, "The result should be different from the input string."
        
        # The output should still be a string
        assert isinstance(result, str), "The result should be a string."
        
        # Check if the output contains all the original characters
        for char in valid_string:
            assert valid_string.count(char) == result.count(char), f"The character '{char}' should appear the same number of times in both strings."
    except InvalidInputError:
        assert False, "shuffle() raised InvalidInputError unexpectedly!"

    # Now test with invalid input
    try:
        shuffle(123)  # Passing an integer should raise InvalidInputError
        assert False, "Expected InvalidInputError for non-string input."
    except InvalidInputError:
        pass  # This is expected

# Run the test
test_shuffle()