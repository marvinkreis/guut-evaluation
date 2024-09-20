from string_utils.validation import is_palindrome

def test__is_palindrome():
    """The mutation changed the palindrome check from != to ==, causing false negatives for valid palindromes."""
    assert is_palindrome("madam") is True, "Should detect 'madam' as a palindrome"
    assert is_palindrome("apple") is False, "Should detect 'apple' as not a palindrome"