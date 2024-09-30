from string_utils.validation import is_palindrome

def test__is_palindrome():
    """
    Test whether the is_palindrome function accurately detects palindromes with case sensitivity.
    The input 'Lol' should return false for the original code, but true if the mutant's ignore_case parameter is true.
    This case will help demonstrate the change in behavior introduced by the mutant.
    """
    output = is_palindrome('Lol')
    assert output == False