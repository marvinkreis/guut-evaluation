from string_utils.validation import is_palindrome

def test_is_palindrome_mutant_killing():
    """
    Test the is_palindrome function with a palindromic input.
    The baseline is expected to return True, while the mutant will 
    raise an IndexError due to incorrect index handling.
    """
    output = is_palindrome("racecar")
    assert output == True, f"Expected True, got {output}"