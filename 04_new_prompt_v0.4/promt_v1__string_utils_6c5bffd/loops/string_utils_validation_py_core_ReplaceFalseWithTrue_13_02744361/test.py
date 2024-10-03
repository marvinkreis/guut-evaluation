from string_utils.validation import is_palindrome

def test__is_palindrome_with_spaces():
    """
    Test whether the is_palindrome function correctly distinguishes between strings that are palindromes 
    when spaces are ignored and those that are not. The input 'race car' should return False for the 
    baseline and True for the mutant, as the mutant ignores spaces by default.
    """
    input_string = "race car"
    output = is_palindrome(input_string)
    assert output is False  # Expect False for baseline, True for mutant