from string_utils.validation import is_palindrome

def test__is_palindrome_case_insensitive():
    """
    Test if the function correctly identifies 'A' as a palindrome when ignoring case.
    The baseline should return True for both 'A' (case-insensitive) and 'A' (case-sensitive),
    while the mutant is expected to return False for the case-insensitive check due to the changed comparison.
    """
    # Case insensitive check
    output_case_insensitive = is_palindrome('A', ignore_case=True)
    assert output_case_insensitive == True  # Expect True on baseline

    # Case sensitive check
    output_case_sensitive = is_palindrome('A')
    assert output_case_sensitive == True  # Also expect True 