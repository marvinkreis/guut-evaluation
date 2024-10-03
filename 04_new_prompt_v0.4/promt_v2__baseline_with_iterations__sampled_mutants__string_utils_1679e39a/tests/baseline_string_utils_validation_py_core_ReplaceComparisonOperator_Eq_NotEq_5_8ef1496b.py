from string_utils.validation import is_isogram

def test__is_isogram():
    """
    Test whether the input string 'dermatoglyphics' is interpreted correctly as an isogram.
    The input string has all unique characters, so it should return true for is_isogram.
    When executed with the mutant, it will return false because the mutant negates the equality check.
    """
    output = is_isogram('dermatoglyphics')
    assert output is True