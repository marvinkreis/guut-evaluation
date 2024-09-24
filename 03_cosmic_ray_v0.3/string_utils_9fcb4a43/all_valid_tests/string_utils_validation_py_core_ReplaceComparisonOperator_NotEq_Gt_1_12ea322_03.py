from string_utils.validation import is_palindrome

def test__is_palindrome_mixed_case():
    """Changing '!=' to '>' in is_palindrome would cause it to incorrectly identify palindromes."""
    input_string = 'Aibohphobia'
    # The correct implementation should return True for this palindrome.
    assert is_palindrome(input_string, ignore_case=True), "Expected to identify 'Aibohphobia' as a palindrome"