from string_utils.validation import is_pangram

def test__is_pangram():
    # Test case 1: A known pangram
    assert is_pangram('The quick brown fox jumps over the lazy dog') == True, "The string should be identified as a pangram."

    # Test case 2: An empty string (should not be a pangram)
    assert is_pangram('') == False, "An empty string should not be identified as a pangram."