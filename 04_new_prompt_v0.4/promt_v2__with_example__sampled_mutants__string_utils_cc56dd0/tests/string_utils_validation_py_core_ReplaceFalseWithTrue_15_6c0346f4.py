from string_utils.validation import is_palindrome

def test_is_palindrome_mutant_killing():
    """
    Test the is_palindrome function with None as input. The mutant incorrectly returns True,
    while the baseline correctly returns False. This identifies that the mutant does not handle
    non-string inputs properly.
    """
    output = is_palindrome(None)
    assert output == False, f"Expected False, got {output}"