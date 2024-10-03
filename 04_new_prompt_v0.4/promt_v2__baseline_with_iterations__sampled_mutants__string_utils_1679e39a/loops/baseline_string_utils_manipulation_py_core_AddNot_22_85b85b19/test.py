from string_utils.manipulation import strip_html
from string_utils.errors import InvalidInputError

def test__strip_html():
    """
    Test that a non-string input raises an InvalidInputError. The mutant's condition 
    allows non-string inputs due to the change from `not is_string(input_string)` to 
    `not not is_string(input_string)`. Hence, passing an integer (which is invalid) 
    will raise an error in the original code but not in the mutant.
    """
    try:
        output = strip_html(12345)  # non-string input
    except InvalidInputError:
        return  # Test passes, function raised the error as expected
    except Exception as e:
        # In case a different exception occurs, the test fails, indicating a problem
        assert False, f"Unexpected exception raised: {e}"
    # If no exception was raised, the test fails
    assert False, "InvalidInputError was not raised as expected."