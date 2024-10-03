from string_utils.validation import contains_html
from string_utils.errors import InvalidInputError  # Ensure the InvalidInputError is imported

def test__contains_html_with_non_string_input():
    """
    Test to verify that the function raises an InvalidInputError when a non-string input is provided.
    This is expected in the Baseline but should not trigger an error in the Mutant due to the change in logic.
    """
    try:
        contains_html(123)  # Passing an integer
        assert False, "Expected an InvalidInputError but did not get one."  # Should raise error in Baseline
    except InvalidInputError:
        pass  # This is the expected behavior in the Baseline
    except TypeError:
        assert False, "TypeError should not occur in Baseline."  # This should not happen in Baseline
    except Exception as e:
        assert False, f"Unexpected exception: {e}"  # Any other exception should fail the test