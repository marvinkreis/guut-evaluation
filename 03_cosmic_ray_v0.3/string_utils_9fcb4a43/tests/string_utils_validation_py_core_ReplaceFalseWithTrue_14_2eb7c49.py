from string_utils.validation import is_palindrome

def test__is_palindrome():
    """The mutant changes the ignore_case parameter to True by default, causing it to incorrectly detect palindromes."""
    output = is_palindrome('Lol')
    assert output is False, "is_palindrome should be case-sensitive and return False for 'Lol'"