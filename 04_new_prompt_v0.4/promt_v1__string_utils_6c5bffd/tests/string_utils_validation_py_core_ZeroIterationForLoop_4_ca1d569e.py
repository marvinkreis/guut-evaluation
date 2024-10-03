from string_utils.validation import is_palindrome

def test__is_palindrome():
    """
    Test the is_palindrome function to confirm it correctly identifies palindromes.
    The test checks two cases:
    1. 'racecar' - which is a palindrome and should return True.
    2. 'hello' - which is not a palindrome and should return False.
    
    The mutant will incorrectly return True for 'hello', while the baseline will return False,
    thereby demonstrating that the mutant is faulty.
    """
    input1 = "racecar"
    input2 = "hello"
    
    output1 = is_palindrome(input1)
    output2 = is_palindrome(input2)

    assert output1 is True, "Expected 'racecar' to be identified as a palindrome."
    assert output2 is False, "Expected 'hello' to be identified as not a palindrome."