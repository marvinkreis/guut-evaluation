from string_utils.validation import is_decimal

def test_is_decimal_mutant_killing():
    """
    Test the is_decimal function with a non-decimal input. 
    The baseline should return False for the input '42', 
    while the mutant will incorrectly return True due to
    the change in logic from 'and' to 'or'.
    """
    output = is_decimal("42")
    assert output == False, f"Expected False, got {output}"