from string_utils.validation import is_palindrome

def test__is_palindrome():
    """
    Test whether the function correctly identifies a palindrome. The input ' ' (a single space) should
    return False in the baseline as it is not a full string. However, when the mutant is present, it 
    will incorrectly return True due to treating non-full strings as valid input, thus exposing the mutant.
    """
    output = is_palindrome(' ')
    assert output == False