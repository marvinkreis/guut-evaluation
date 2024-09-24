from string_utils.validation import is_palindrome

def test__is_palindrome():
    """Changing '!=' to '>' in is_palindrome would cause it to incorrectly identify palindromes."""
    input_string = 'A man a plan a canal Panama'
    assert is_palindrome(input_string, ignore_spaces=True, ignore_case=True), "Expected to identify a palindrome"