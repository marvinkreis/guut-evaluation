from string_utils.validation import is_palindrome

def test_is_palindrome():
    # Test case that should return True when ignoring spaces and case
    assert is_palindrome('A man a plan a canal Panama', ignore_spaces=True, ignore_case=True) == True, "Expected True; should allow for palindrome when ignoring spaces and case."

    # Test case that should return False when spaces should matter (default behavior with no ignores)
    assert is_palindrome('A man a plan a canal Panama', ignore_spaces=False, ignore_case=False) == False, "Expected False; spaces do matter in this check."

    # Test case that should return True, ignoring spaces and case (No lemon, no melon)
    assert is_palindrome('No lemon no melon', ignore_spaces=True, ignore_case=True) == True, "Expected True; should recognize as a palindrome ignoring spaces and case."

    # Test case that should return False when case matters but ignores spaces (testing behavior)
    assert is_palindrome('No lemon no melon', ignore_spaces=False, ignore_case=True) == False, "Expected False; spaces should not allow for palindrome even if case is ignored."

    # A single character is always a palindrome
    assert is_palindrome('A', ignore_spaces=False, ignore_case=False) == True, "Expected True; single character is always a palindrome."

    # Edge case with only spaces (should return False regardless)
    assert is_palindrome('       ', ignore_spaces=False, ignore_case=False) == False, "Expected False; only spaces are not palindromic."

    # A new test case that should fail with the mutant
    # When ignore_spaces is True or Default is True, it should return True
    assert is_palindrome('Able was I saw Elba', ignore_spaces=True, ignore_case=True) == True, "Expected True; should allow for palindrome when ignoring spaces and case."
    
    # When ignore_spaces is False and ignore_case is False, it should return False
    assert is_palindrome('Able was I saw Elba', ignore_spaces=False, ignore_case=False) == False, "Expected False; spaces matter and should not allow palindrome."