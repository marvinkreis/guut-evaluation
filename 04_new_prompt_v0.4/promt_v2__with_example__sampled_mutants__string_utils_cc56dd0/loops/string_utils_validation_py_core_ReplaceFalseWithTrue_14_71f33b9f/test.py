from string_utils.validation import is_palindrome

def test_is_palindrome_mutant_killing():
    """
    Test the is_palindrome function with 'Level'. The mutant will ignore case
    differences and return True, while the baseline should return False.
    """
    output = is_palindrome("Level")
    assert output == False, f"Expected False, but got {output}"