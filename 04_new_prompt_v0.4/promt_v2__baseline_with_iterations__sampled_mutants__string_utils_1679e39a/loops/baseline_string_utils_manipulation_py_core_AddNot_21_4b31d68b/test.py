from string_utils.manipulation import shuffle
from string_utils.errors import InvalidInputError

def test__shuffle():
    """
    Test whether the shuffle function raises an InvalidInputError when given non-string input.
    The expected behavior is that an InvalidInputError should be raised when input is not a string.
    The mutant changes the validity check to allow non-string types, so this test will fail with the mutant.
    """
    try:
        shuffle(123)  # passing an integer instead of a string should raise an error
    except InvalidInputError as e:
        assert str(e) == 'Expected "str", received "int"'
    else:
        raise AssertionError("Expected InvalidInputError was not raised.")