from string_utils.validation import is_credit_card

def test_is_credit_card_mutant_killing():
    """
    Test the is_credit_card function with a valid credit card number.
    The mutant will incorrectly return False, while the baseline returns True.
    This distinguishes the mutant from the baseline behavior.
    """
    output = is_credit_card("4111111111111111")
    assert output == True, f"Expected True, got {output}"