from string_utils.validation import is_palindrome

def test__is_palindrome_case_insensitive():
    # This should return True as 'Racecar' is a palindrome ignoring case
    assert is_palindrome('Racecar', ignore_case=True) == True
    # This should return False as 'Racecar' is not a palindrome considering case
    assert is_palindrome('Racecar', ignore_case=False) == False