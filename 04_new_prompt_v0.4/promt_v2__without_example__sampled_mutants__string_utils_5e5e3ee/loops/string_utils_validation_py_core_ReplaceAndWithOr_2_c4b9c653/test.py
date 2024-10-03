from string_utils.validation import is_decimal

def test__is_decimal_mutant_killing():
    """
    Test if the is_decimal function can correctly identify numbers that are
    not decimals. The input '42' should return False, while '42.0' should return True.
    This test aims to highlight the faulty behavior of the mutant, which incorrectly
    identifies '42' as a decimal.
    """
    assert is_decimal('42') == False  # This should fail with the mutant
    assert is_decimal('42.0') == True  # This should pass on both