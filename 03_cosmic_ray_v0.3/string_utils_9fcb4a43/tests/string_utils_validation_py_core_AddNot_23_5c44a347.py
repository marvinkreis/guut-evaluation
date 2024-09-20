from string_utils.validation import is_palindrome

def test__is_palindrome():
    """The mutant will incorrectly return False when checking a case-insensitive palindrome."""
    assert is_palindrome('Lol', ignore_case=True), "Expected 'Lol' to be a palindrome when ignoring case."