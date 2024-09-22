from string_utils.validation import is_palindrome

def test_is_palindrome():
    # Case where input is an empty string (should be false for palindrome).
    assert not is_palindrome('')  # Expected: False (correct behavior)
    
    # Case where input is None (should be false for palindrome).
    assert not is_palindrome(None)  # Expected: False (correct behavior)
    
    # Case where input is a single character (should be true for palindrome).
    assert is_palindrome('a')  # Expected: True (correct behavior)
    
    # Test with a multi-character palindrome ignoring case and spaces.
    assert is_palindrome('A man a plan a canal Panama'.replace(' ', '').lower())  # Expected: True (correct behavior)

    # Case with a non-palindrome.
    assert not is_palindrome('Hello World')  # Expected: False (correct behavior)

    # Edge case: with spaces not ignored (should return false).
    assert not is_palindrome('A man a plan a canal Panama', ignore_spaces=False)  # Expected: False (correct behavior)

# Execute the test
test_is_palindrome()