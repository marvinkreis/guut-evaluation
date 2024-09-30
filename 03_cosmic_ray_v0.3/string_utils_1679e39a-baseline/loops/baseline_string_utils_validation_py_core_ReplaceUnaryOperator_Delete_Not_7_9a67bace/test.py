from string_utils.validation import is_palindrome

def test__is_palindrome():
    """
    Test to check if the function correctly identifies a palindrome when the 
    input string is 'racecar'. The mutant modifies the check to invert the 
    condition, causing it to return True for non-full strings instead of False.
    This test will pass with the original code because 'racecar' is a palindrome,
    but will fail with the mutant.
    """
    output = is_palindrome('racecar')
    assert output == True