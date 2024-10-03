from string_utils.manipulation import booleanize
from string_utils.errors import InvalidInputError

def test_booleanize_mutant_killing():
    """
    Test the booleanize function with a non-string input.
    The baseline will raise an InvalidInputError, while the mutant will raise an AttributeError due to flawed input checking.
    """
    try:
        booleanize(123)
        assert False, "Expected an error for non-string input"
    except InvalidInputError as e:
        print(f"Baseline raised InvalidInputError: {e}")
    except AttributeError as e:
        print(f"Mutant raised AttributeError as expected: {e}")
        assert False, "Mutant behavior detected - should not have raised this error"