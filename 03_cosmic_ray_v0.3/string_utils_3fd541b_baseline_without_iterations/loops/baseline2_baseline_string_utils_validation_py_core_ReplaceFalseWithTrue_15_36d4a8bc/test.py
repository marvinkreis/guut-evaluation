from string_utils.validation import is_palindrome

def test__is_palindrome():
    # Test with a valid palindrome
    assert is_palindrome('A man a plan a canal Panama', ignore_spaces=True, ignore_case=True) == True
    # Test with a non-palindrome string
    assert is_palindrome('hello') == False
    # Test with an empty string (should be considered a valid palindrome)
    assert is_palindrome('') == False  # This should return False with the original code, True with the mutant
    # Test with a string that is a palindrome ignoring spaces and case
    assert is_palindrome('Was it a car or a cat I saw', ignore_spaces=True, ignore_case=True) == True