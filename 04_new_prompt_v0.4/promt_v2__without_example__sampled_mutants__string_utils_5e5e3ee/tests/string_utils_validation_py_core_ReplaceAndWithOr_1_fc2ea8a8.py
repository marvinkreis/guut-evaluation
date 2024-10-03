from string_utils.validation import is_integer

def test__is_integer_mutant_killing():
    """
    This test checks whether the function correctly identifies non-integer decimal strings. 
    The input '3.14' should not be considered an integer. 
    The baseline should return False, while the mutant is expected to return True.
    """
    result = is_integer('3.14')
    assert result == False, "Expected output is False for the baseline."