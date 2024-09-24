from string_utils.validation import is_palindrome

def test_palindrome_detection():
    # Palindrome tests (should return True)
    assert is_palindrome("madam", ignore_case=True)  # Simple palindrome
    assert is_palindrome("racecar", ignore_case=True)  # Simple palindrome
    assert is_palindrome("A man a plan a canal Panama", ignore_case=True, ignore_spaces=True)  # Palindrome with spaces
    assert is_palindrome("Able was I ere I saw Elba", ignore_case=True, ignore_spaces=True)  # Palindrome
    assert is_palindrome("No lemon, no melon", ignore_case=True, ignore_spaces=True)  # Palindrome
    assert is_palindrome("abcba")  # Simple alphanumeric palindrome
    assert is_palindrome("abcdeedcba")  # Correctly identified as a palindrome
    assert is_palindrome("A")  # Single character (is a palindrome)
    assert is_palindrome("B")  # Single character (is a palindrome)

    # Non-palindrome tests (should return False)
    assert not is_palindrome("Hello")  # Not a palindrome (case sensitive)
    assert not is_palindrome("world")  # Not a palindrome
    assert not is_palindrome("abcdef")  # Not a palindrome
    assert not is_palindrome("This is not a palindrome")  # Clear non-palindrome
    assert not is_palindrome("ab")  # Two different characters (not a palindrome)
    assert not is_palindrome("palindrome!")  # Not a palindrome due to punctuation

    # Specific tests to catch mutant behavior 
    # These are crafted to trick the mutant's logic
    assert is_palindrome("ABccBA", ignore_case=True)  # Should be true (palindrome regardless of case)
    assert not is_palindrome("AABccBA")  # Should be False due to extra 'A'
    
    # Introducing tricky cases
    assert not is_palindrome("abcdeedcb")  # Should be false, one character mismatch, mutant logic might fail here
    assert is_palindrome("abccba")  # Should still be true
    assert not is_palindrome("abcdefg")  # Clear non-palindrome
    assert is_palindrome("abccba")  # This is still true
    assert not is_palindrome("abcdeedca")  # Should be false due to mismatched characters
    
    # Introducing a tricky non-palindrome with repeating characters
    assert not is_palindrome("aabbccdde")  # Should be false, but mutant might catch it wrongly
    assert is_palindrome("aaabaaa")  # This is correctly a palindrome

    print("All assertions passed!")  # Indicate all checks have run

# Running the test function to execute
test_palindrome_detection()