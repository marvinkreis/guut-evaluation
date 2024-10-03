from string_utils.validation import is_palindrome

def test__is_palindrome():
    """
    Test the is_palindrome function to verify correct behavior for known palindromes.
    The input 'madam' is a palindrome and should return True. The mutant alters the index logic,
    leading to an IndexError, differentiating it from the baseline.
    """
    input_string = "madam"
    output = is_palindrome(input_string)
    assert output == True