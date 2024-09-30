from string_utils.validation import is_palindrome

def test__is_palindrome():
    """
    Test whether the palindrome function correctly identifies a case-insensitive palindrome.
    The input 'Racecar' should return true since it's a palindrome ignoring case,
    but the mutant version incorrectly uses 'is not' instead of '!=' for character comparison,
    leading to a false result when it should be true.
    """
    output = is_palindrome('Racecar', ignore_case=True)
    assert output == True  # This assert should pass with the baseline but fail with the mutant.