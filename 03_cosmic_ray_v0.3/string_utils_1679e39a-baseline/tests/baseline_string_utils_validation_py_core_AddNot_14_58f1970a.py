from string_utils.validation import is_credit_card

def test__is_credit_card():
    """
    This test checks whether the function correctly processes a valid credit card type.
    The input is a valid credit card number with a valid card type.
    The mutant logic will incorrectly allow an invalid card type, leading to a different 
    outcome compared to the original function.
    """
    # Using a valid card type for the test; this should pass the test under normal circumstances.
    output = is_credit_card('4111111111111111', card_type='VISA')
    assert output == True  # This should be true for a valid credit card.