from string_utils.validation import is_pangram

def test__is_pangram():
    """Changing the return value in is_pangram when input is not a full string would cause it to incorrectly identify non-full strings as pangrams."""
    
    # Test with empty string
    assert not is_pangram(""), "Expected False for empty string"
    
    # Test with string of spaces
    assert not is_pangram("    "), "Expected False for string with only spaces"