from string_utils.validation import is_palindrome

def test__is_palindrome():
    # Test for a known palindrome
    palindrome_input = "A man a plan a canal Panama"
    assert is_palindrome(palindrome_input, ignore_spaces=True, ignore_case=True) == True, "The input should be detected as a palindrome."

    # Test for a non-palindrome
    non_palindrome_input = "hello"
    assert is_palindrome(non_palindrome_input) == False, "The input should be detected as not a palindrome."