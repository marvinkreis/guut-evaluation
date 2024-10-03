from string_utils.manipulation import booleanize
from string_utils.errors import InvalidInputError  # Importing the exception class

def test__booleanize():
    """
    Test the booleanize function with a negative case. The input '123' simulates an invalid string 
    because it's numeric and should raise an InvalidInputError in the correct code. The mutant 
    changes the condition and will not raise an error for invalid input. Therefore, using an 
    integer input will cause a different behavior on the mutant.
    """
    try:
        booleanize(123)  # this should raise an InvalidInputError
        assert False  # If no error is raised, the test fails
    except InvalidInputError:
        pass  # The error indicates the code is correct; test passes