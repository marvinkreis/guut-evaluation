from string_utils.validation import is_full_string

def test__is_full_string():
    # Test cases that should return true
    assert is_full_string('hello') == True
    assert is_full_string('   world   ') == True
    
    # Test cases that should return false
    assert is_full_string('') == False
    assert is_full_string('    ') == False
    assert is_full_string(None) == False
    assert is_full_string(123) == False  # Non-string input
    
    # Additionally test cases that the mutant would break
    assert is_full_string(' ') == False  # was supposed to be False but may pass mutant condition