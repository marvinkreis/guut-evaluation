from string_utils.manipulation import shuffle
from string_utils.errors import InvalidInputError

def test__shuffle_invalid_input_kills_mutant():
    """
    Test that invalid input raises an InvalidInputError in the Baseline,
    but causes a TypeError in the Mutant due to incorrect input validation.
    This test confirms that the mutant introduces a logic error by allowing invalid input.
    """
    # Testing invalid integer input
    try:
        shuffle(123)  # Invalid: integer instead of string
        # If no exception is raised, the mutant allows invalid input: test should fail here.
        assert False, "Expected InvalidInputError for integer input, but none was raised."
    except InvalidInputError:
        pass  # Correct: expected behavior for baseline.
    
    # Checking mutant behavior for the integer input:
    # This will allow us to observe the mutant behavior in a separate block.
    try:
        shuffle(123)  # Invalid: integer instead of string
    except Exception as e:
        print(f"Mutant output for integer input raised an exception: {e}")

    # Testing invalid None input
    try:
        shuffle(None)  # Invalid: None instead of string
        # If no exception is raised, the mutant allows invalid input: test should fail here.
        assert False, "Expected InvalidInputError for None input, but none was raised."
    except InvalidInputError:
        pass  # Correct: expected behavior for baseline.
    
    # Checking mutant behavior for the None input:
    # This will allow us to observe the mutant behavior in a separate block.
    try:
        output = shuffle(None)  # Invalid: None instead of string
    except Exception as e:
        print(f"Mutant output for None input raised an exception: {e}")