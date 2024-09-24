from string_utils.validation import is_isogram

def test_is_isogram():
    # Valid isogram cases; True for both versions
    assert is_isogram("abcdefg") == True  # Unique 
    assert is_isogram("a") == True  # Unique 

    # Invalid isogram cases; should return False for both
    assert is_isogram("hello") == False  # 'l' repeats
    assert is_isogram("aabbcc") == False  # All characters repeating
    assert is_isogram("aaaa") == False  # All characters are the same
    assert is_isogram("abcabc") == False  # Full duplicate
    assert is_isogram("xyzy") == False  # 'y' repeats
    assert is_isogram("abab") == False  # Multiple repeats
    
    # Testing with guaranteed duplication leading to a clear False
    assert is_isogram("bcdeb") == False  # 'b' repeats
    assert is_isogram("abcdefgg") == False  # 'g' repeats
    assert is_isogram("xxyy") == False  # 'x' and 'y' repeats

    # Confirm edge case with empty string
    assert is_isogram("") == False  # Empty string must return False

    # All unique cases to remain True
    assert is_isogram("dermatoglyphics") == True  # All unique

    # These cases should focus especially on detecting the mutant
    assert is_isogram("abcdeeffg") == False  # 'e' repeats
    assert is_isogram("abcdefghijkna") == False  # 'a' repeats

    print("All tests passed!")