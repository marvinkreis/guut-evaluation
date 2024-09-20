from string_utils.validation import is_palindrome

def test__is_palindrome():
    """The mutant would incorrectly return True for non-full strings."""
    assert is_palindrome(None) is False, "Expected False for None input"
    assert is_palindrome('') is False, "Expected False for an empty string"
    assert is_palindrome(' ') is False, "Expected False for whitespace input"