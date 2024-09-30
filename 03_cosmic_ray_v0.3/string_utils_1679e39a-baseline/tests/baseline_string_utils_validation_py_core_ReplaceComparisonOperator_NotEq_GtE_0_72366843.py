from string_utils.validation import is_full_string

def test__is_full_string():
    """
    Test whether a string containing only whitespace is correctly identified as not being a full string.
    The input is a string with just a space character, which should return False for is_full_string.
    The mutant incorrectly uses a comparison (>=) that would evaluate this input incorrectly.
    """
    output = is_full_string(' ')
    assert output == False