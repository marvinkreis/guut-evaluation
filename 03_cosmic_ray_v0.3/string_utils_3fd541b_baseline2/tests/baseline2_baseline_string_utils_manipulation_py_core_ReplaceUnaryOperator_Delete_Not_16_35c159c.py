from string_utils.manipulation import booleanize
from string_utils.errors import InvalidInputError

def test_booleanize_invalid_input():
    try:
        # This should raise an InvalidInputError because the input is not a string
        booleanize(123)  # Using an integer as input
    except InvalidInputError:
        # Test passed; the error was correctly raised
        return
    except Exception as e:
        # Any other exception means the test did not pass
        assert False, f"Unexpected exception raised: {e}"
    
    # If no exception was raised, the test should fail
    assert False, "Expected InvalidInputError not raised."