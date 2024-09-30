from string_utils.validation import is_palindrome

def test__is_palindrome():
    """
    Test whether the function correctly identifies a simple palindrome without ignoring spaces or case. 
    The input 'A man a plan a canal Panama' would be assessed differently if the mutant erroneously modifies 
    the index calculation, leading to an incorrect result. The correct implementation expects 'A man a plan a canal Panama'
    to evaluate as false when case and spaces are considered, but the mutant could return true.
    """
    output = is_palindrome('A man a plan a canal Panama')  # Should return False
    assert output == False