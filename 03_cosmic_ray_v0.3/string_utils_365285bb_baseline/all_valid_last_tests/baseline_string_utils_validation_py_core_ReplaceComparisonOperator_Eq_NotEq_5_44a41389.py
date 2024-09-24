from string_utils.validation import is_isogram

def test_is_isogram():
    # Valid isogram cases should return True
    assert is_isogram('dermatoglyphics') == True  # An isogram
    assert is_isogram('abcde') == True            # An isogram with unique letters

    # Invalid isogram cases should return False
    assert is_isogram('hello') == False           # Duplicate 'l' makes it non-isogram
    assert is_isogram('world') == True            # An isogram

    # Edge cases
    assert is_isogram('') == False                 # An empty string is not an isogram
    assert is_isogram('a') == True                 # A single character string is an isogram
    
    # Additional test to ensure mutant is detected
    assert is_isogram('abcabc') == False           # Not an isogram, should return False for correct code
