from string_utils.manipulation import strip_margin
from string_utils.errors import InvalidInputError

def test__strip_margin_non_string_input():
    try:
        # Attempt to call strip_margin with a non-string input (e.g., an integer)
        strip_margin(42)
        # If no exception is raised, the test should fail
        assert False, "Expected InvalidInputError to be raised"
    except InvalidInputError:
        # Test passes as we expect an InvalidInputError to be raised
        pass
    except Exception as e:
        # If any other exception is raised, fail the test
        assert False, f"Unexpected exception raised: {str(e)}"