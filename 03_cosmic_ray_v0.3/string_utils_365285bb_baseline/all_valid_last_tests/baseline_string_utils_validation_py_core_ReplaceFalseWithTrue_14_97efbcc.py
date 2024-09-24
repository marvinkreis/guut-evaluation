from string_utils.validation import is_palindrome

def test_is_palindrome():
    # This test checks if the function correctly identifies case sensitivity
    
    # The original should return False due to strict case, while the mutant returns True
    assert is_palindrome('Racecar', ignore_spaces=False, ignore_case=False) == False  # Original: False, Mutant: True
    assert is_palindrome('racecar', ignore_spaces=False, ignore_case=False) == True   # Both return True

    # Full phrase that is a clear palindrome
    assert is_palindrome('A man a plan a canal Panama', ignore_spaces=False, ignore_case=False) == False  # Original: False, Mutant: True
    assert is_palindrome('A man a plan a canal Panama', ignore_spaces=True, ignore_case=False) == False   # Original: False, Mutant: True

    # Checking case sensitivity and spaces handled properly
    assert is_palindrome('A man a plan a canal Panama', ignore_spaces=True, ignore_case=True) == True    # Both should return True

    # Let's use the same phrase without punctuation for a palindrome check
    assert is_palindrome('No lemon no melon', ignore_spaces=True, ignore_case=True) == True  # Both should return True
    assert is_palindrome('No lemon no melon', ignore_spaces=False, ignore_case=False) == False  # Original: False, Mutant: True

    # Punctuation handling check
    assert is_palindrome('No lemon, no melon', ignore_spaces=True, ignore_case=True) == True  # Both should return True
    assert is_palindrome('No lemon, no melon', ignore_spaces=True, ignore_case=False) == False # Original: False, Mutant: True

    # Non-palindrome tests to ensure control
    assert is_palindrome('Hello World', ignore_spaces=True, ignore_case=False) == False   # Both should return False
    assert is_palindrome('hello world', ignore_spaces=True, ignore_case=True) == False   # Both should return False