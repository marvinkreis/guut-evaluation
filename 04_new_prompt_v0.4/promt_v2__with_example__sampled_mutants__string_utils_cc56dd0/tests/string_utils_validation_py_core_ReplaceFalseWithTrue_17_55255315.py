from string_utils.validation import is_pangram

def test_is_pangram_mutant_killing():
    """
    Test the is_pangram function with various edge cases including an empty string
    and a string that contains only whitespace. The mutant will return True for these
    cases, while the baseline will return False.
    """
    empty_output = is_pangram('')
    whitespace_output = is_pangram('   ')
    
    assert empty_output == False, f"Expected False for empty string, got {empty_output}"
    assert whitespace_output == False, f"Expected False for whitespace string, got {whitespace_output}"