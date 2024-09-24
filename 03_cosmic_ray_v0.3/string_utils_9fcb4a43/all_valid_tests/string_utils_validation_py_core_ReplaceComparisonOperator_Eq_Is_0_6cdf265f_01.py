from string_utils.validation import is_isogram

def test__is_isogram():
    """Test cases to verify that the mutant implementation behaves incorrectly."""
    
    # Valid isogram
    assert is_isogram("abcdefg") == True, "Expected to be an isogram"
    
    # Two repeated characters, should return False
    assert is_isogram("aa") == False, "Expected not to be an isogram"
    
    # Longer valid isogram
    assert is_isogram("abcdefghij") == True, "Expected to be an isogram"
    
    # Check long repeated characters, should return False
    assert is_isogram("a" * 1000) == False, "Expected not to be an isogram"

    # Long repeating characters should return False
    assert is_isogram("abcd" * 250) == False, "Expected not to be an isogram"