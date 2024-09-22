from string_utils.validation import is_pangram

def test_is_pangram():
    # Test case where input is not a full string
    result = is_pangram(None)  # Should return False in the original code
    assert result == False, f"Expected False, got {result}"

    # Test case with a valid pangram
    result = is_pangram('The quick brown fox jumps over the lazy dog')  # Should return True
    assert result == True, f"Expected True, got {result}"

    # Test case with a known non-pangram
    result = is_pangram('hello world')  # Should return False
    assert result == False, f"Expected False, got {result}"

    # Test case where the input is an empty string
    result = is_pangram('')  # Should return False
    assert result == False, f"Expected False, got {result}"