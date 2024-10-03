from string_utils.manipulation import shuffle
from string_utils.errors import InvalidInputError  # Import the custom error

def test_shuffle_invalid_input():
    """
    This test checks that the shuffle function raises InvalidInputError for
    non-string inputs. The baseline should raise InvalidInputError, while 
    the mutant should lead to a TypeError when processing a non-string input.
    """
    # Test the baseline behavior
    try:
        shuffle(12345)
        assert False, "Expected InvalidInputError, but no error was raised."
    except InvalidInputError:
        pass  # This is the expected behavior for the baseline

    # Test the mutant behavior
    try:
        output = shuffle(12345)
        assert False, "Expected TypeError, but no error was raised."  # This should happen in mutant
    except TypeError:
        pass  # This is the expected behavior from the mutant