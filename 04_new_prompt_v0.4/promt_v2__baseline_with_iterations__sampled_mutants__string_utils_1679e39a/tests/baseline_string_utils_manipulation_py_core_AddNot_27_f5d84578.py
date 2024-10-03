from string_utils.manipulation import strip_margin
from string_utils.errors import InvalidInputError

def test__strip_margin():
    """
    Test that the function correctly raises an InvalidInputError when the input is not a string.
    The mutant changes the logic in the condition to check if the input is NOT a string.
    Therefore, passing a non-string input should raise an error in the correct code (baseline),
    while the mutant will not raise an error, allowing us to detect the mutant.
    """
    try:
        strip_margin(12345)  # passing an integer instead of a string
    except InvalidInputError:
        # This is expected, meaning the baseline works correctly
        return  # The test passes in baseline
    except Exception as e:
        assert False, f"Expected InvalidInputError, but got {type(e).__name__}"
    
    # If no exception was raised, the test fails.
    assert False, "Expected an InvalidInputError but none was raised."