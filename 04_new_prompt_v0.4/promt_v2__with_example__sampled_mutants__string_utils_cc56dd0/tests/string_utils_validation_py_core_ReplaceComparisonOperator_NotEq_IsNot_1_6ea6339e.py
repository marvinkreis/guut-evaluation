from string_utils.validation import is_palindrome

def test_is_palindrome_mutant_killing():
    """
    Test the is_palindrome function using the input 'Able was I saw Elba'. 
    The baseline will recognize this string as a palindrome and return True, while 
    the mutant will fail to do so and return False due to the change from '!=' 
    to 'is not' in character comparison.
    """
    output = is_palindrome("Able was I saw Elba", ignore_case=True)
    assert output == True, f"Expected True, got {output}"