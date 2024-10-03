from string_utils.validation import is_credit_card

def test_is_credit_card_mutant_killing():
    """
    Test the is_credit_card function using an invalid credit card number.
    The mutant will return True for an invalid card, while the baseline will return False.
    """
    output = is_credit_card("1234567890123456")
    assert output == False, f"Expected False, got {output}"