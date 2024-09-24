import re

# This regex correctly captures uppercase letters following punctuation
UPPERCASE_AFTER_SIGN_REGEX = re.compile(r'([.?!]\s)([A-Z])', re.MULTILINE | re.UNICODE)

def test_UPPERCASE_AFTER_SIGN():
    # This should match because '.' is followed by a space and 'A'
    test_string1 = "This is a test. A valid uppercase after sign."
    
    # Should not match because '?' is followed directly by lowercase
    test_string2 = "What is this? a lowercase after question mark."

    # This should match because '!' is followed by a space and 'A'
    test_string3 = "Great! A stunning performance!"
    
    # Should not match due to the spacing issue (missing space before the 'A')
    test_string4 = "Incredible!Amazing people!"

    # This should match (correctly formatted question followed by uppercase)
    test_string5 = "Did you see that? Yes, indeed!"
    
    # Should not match, '?' followed directly by 'n'
    test_string6 = "Is this right?no, it isn't."

    # This should NOT match due to a missing space (punctuation issue)
    test_string7 = "Well done!excellent!"

    # Matches correct space followed by uppercase
    assert UPPERCASE_AFTER_SIGN_REGEX.search(test_string1) is not None, "Test 1 failed - valid case should match."
    assert UPPERCASE_AFTER_SIGN_REGEX.search(test_string2) is None, "Test 2 failed - invalid case should not match."
    assert UPPERCASE_AFTER_SIGN_REGEX.search(test_string3) is not None, "Test 3 failed - valid case should match."
    assert UPPERCASE_AFTER_SIGN_REGEX.search(test_string4) is None, "Test 4 failed - invalid case should not match due to spacing!"
    assert UPPERCASE_AFTER_SIGN_REGEX.search(test_string5) is not None, "Test 5 failed - valid case should match."
    assert UPPERCASE_AFTER_SIGN_REGEX.search(test_string6) is None, "Test 6 failed - invalid case should not match."
    assert UPPERCASE_AFTER_SIGN_REGEX.search(test_string7) is None, "Test 7 failed - invalid case should not match due to missing space!"

# To execute the test function
if __name__ == "__main__":
    try:
        test_UPPERCASE_AFTER_SIGN()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")