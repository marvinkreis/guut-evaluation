from string_utils.manipulation import booleanize
from string_utils.errors import InvalidInputError

def test__booleanize_invalid_input():
    """
    Test the booleanize function to ensure it raises an InvalidInputError
    when provided with non-string inputs like integers or lists, 
    which behaves correctly in the baseline but incorrectly in the mutant.
    """
    try:
        booleanize(1)  # This should raise an InvalidInputError
        assert False, "Expected InvalidInputError for integer input"
    except InvalidInputError:
        pass  # Expected behavior

    try:
        booleanize(['yes'])  # This should raise an InvalidInputError
        assert False, "Expected InvalidInputError for list input"
    except InvalidInputError:
        pass  # Expected behavior