from string_utils.validation import is_palindrome

def test__is_palindrome_case_difference():
    """
    Test whether the is_palindrome function behaves differently due to the change in default value of ignore_case.
    The input "Lol" should return False under baseline (ignore_case=False) and True under the mutant (ignore_case=True).
    """
    output = is_palindrome("Lol")
    assert output == False  # This should pass for the baseline