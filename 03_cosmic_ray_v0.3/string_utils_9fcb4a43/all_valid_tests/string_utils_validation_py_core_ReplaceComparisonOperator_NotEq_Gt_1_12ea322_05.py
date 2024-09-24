from string_utils.validation import is_palindrome

def test__is_palindrome_edge_cases():
    """Tests to differentiate the correct logic from the mutant logic."""
    
    # Test with a simple known palindrome
    palindrome_input = 'Noon'
    assert is_palindrome(palindrome_input, ignore_case=True), "Expected to identify 'Noon' as a palindrome"
    
    # Test with a non-palindromic string
    non_palindrome_input = 'Hello'
    assert not is_palindrome(non_palindrome_input), "Expected 'Hello' not to be identified as a palindrome"