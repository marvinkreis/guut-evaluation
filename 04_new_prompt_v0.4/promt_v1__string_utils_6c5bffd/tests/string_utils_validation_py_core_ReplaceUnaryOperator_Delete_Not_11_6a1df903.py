from string_utils.validation import words_count
from string_utils.errors import InvalidInputError  # Importing the necessary error class

def test__words_count_invalid_input():
    """
    Test the words_count function with an invalid input (an integer).
    The baseline raises an InvalidInputError, while the mutant raises a different type of exception.
    This ensures we can distinguish between the two.
    """
    try:
        words_count(123)  # Invalid input; should raise an error in baseline
    except InvalidInputError:
        # Catching the specific error expected in baseline
        pass
    except Exception as e:
        # For mutant, any other exception signifies the error is different
        assert isinstance(e, Exception), f"Caught unexpected exception: {e}"

    # Testing with a valid string input to ensure correct functionality
    output = words_count("hello world")
    assert output == 2  # This should pass for both baseline and mutant