from string_utils.manipulation import booleanize
from string_utils.errors import InvalidInputError  # Importing the error class

def test__booleanize_non_string():
    """
    Test booleanize function with a non-string input (e.g. an integer). The base function should raise an InvalidInputError,
    while the mutant allows execution to proceed, leading to an AttributeError. This test is designed to differentiate their behavior.
    """
    try:
        booleanize(123)  # Pass an integer which should trigger an error
        assert False, "Expected InvalidInputError but no exception was raised"  # This should not execute
    except InvalidInputError as e:
        print(f"output = {str(e)}")  # Correctly captured error in Baseline
    except Exception as e:
        # Any other exception by the mutant should be caught and logged.
        assert False, f"Unexpected exception: {str(e)}"  # This will ensure we capture unexpected behavior