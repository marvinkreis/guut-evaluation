from string_utils.validation import is_palindrome

def test__is_palindrome():
    """
    This test checks whether the input string 'A man a plan a canal Panama' is correctly identified as a palindrome.
    The expected output is True, while the mutant will return False due to the improper comparison in the code change.
    """
    input_string = 'A man a plan a canal Panama'
    output = is_palindrome(input_string, ignore_spaces=True, ignore_case=True)
    assert output == True