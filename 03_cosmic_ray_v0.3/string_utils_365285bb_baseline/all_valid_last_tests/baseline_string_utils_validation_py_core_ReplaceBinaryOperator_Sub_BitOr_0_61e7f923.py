from string_utils.validation import is_palindrome

def test_is_palindrome():
    # Test cases for palindrome
    assert is_palindrome('madam') == True  # True for a palindrome
    assert is_palindrome('racecar') == True  # True for a palindrome
    assert is_palindrome('hello') == False  # False for a non-palindrome
    assert is_palindrome('A man a plan a canal Panama', ignore_spaces=True, ignore_case=True) == True  # Palindrome with ignored spaces and case

    # The following input will return different results when tested with the mutant.
    # This string has punctuation and case variations
    assert is_palindrome('Was it a car or a cat I saw?') == False  # This should be False (not a palindrome)

    # This should also return different results with the mutant due to the faulty tail character calculation
    assert is_palindrome('No lemon, no melon', ignore_spaces=True, ignore_case=True) == True  # This should be a True palindrome