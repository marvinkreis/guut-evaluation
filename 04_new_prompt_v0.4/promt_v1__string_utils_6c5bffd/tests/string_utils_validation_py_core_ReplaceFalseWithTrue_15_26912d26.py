from string_utils.validation import is_palindrome

def test__is_palindrome_mutant_killer():
    """
    Test the is_palindrome function specifically with invalid inputs.
    The input None and empty strings should return False on the baseline, 
    but True on the mutant. This test checks for that specific behavior.
    """
    assert is_palindrome(None) == False  # Expecting False on both Baseline and Mutant
    assert is_palindrome('') == False    # Expecting False on both Baseline and Mutant