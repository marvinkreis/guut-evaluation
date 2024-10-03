from string_utils.validation import is_credit_card

def test_is_credit_card_mutant_killing():
    """
    Test the is_credit_card function with a valid VISA credit card number.
    The baseline should return True, but the mutant will return False due to
    the incorrect logic change in the matching condition.
    """
    output = is_credit_card('4111111111111111', card_type='VISA')
    assert output == True, f"Expected True, got {output}"