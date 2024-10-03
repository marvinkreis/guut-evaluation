from string_utils.manipulation import strip_html
from string_utils.errors import InvalidInputError

def test__strip_html_invalid_input():
    """
    Test that strip_html raises InvalidInputError on non-string input for the baseline,
    and that it raises a TypeError for the mutant.
    This effectively distinguishes between the two implementations.
    """

    # Testing on the baseline
    try:
        strip_html(123)  # passing an integer instead of a string
        assert False, "Expected InvalidInputError, but no exception was raised"
    except InvalidInputError:
        pass  # This is expected behavior for the baseline

    # Testing on the mutant
    try:
        strip_html(123)  # passing an integer instead of a string
        assert False, "Expected TypeError, but no exception was raised"
    except TypeError:
        pass  # This is expected behavior for the mutant