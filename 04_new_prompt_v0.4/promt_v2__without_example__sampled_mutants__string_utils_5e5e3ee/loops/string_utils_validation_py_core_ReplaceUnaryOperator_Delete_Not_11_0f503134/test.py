from string_utils.validation import words_count
from string_utils.errors import InvalidInputError

def test__words_count():
    """
    Test words_count function for a valid input and an invalid input.
    The input 'hello world' should return 2 in the baseline,
    while passing None should raise an InvalidInputError only in the baseline,
    not in the mutant where the check is improperly inverted.
    """
    # Test with a valid string input
    assert words_count('hello world') == 2

    # Test with an invalid input (None)
    try:
        words_count(None)
        # If no exception is raised, it should fail the test for the mutant
        assert False, "Expected InvalidInputError for None input."
    except InvalidInputError:
        pass  # Expected to pass in baseline
    except Exception as e:
        # If some other exception is raised, it should also fail the test
        assert False, f"Unexpected exception raised: {e}"