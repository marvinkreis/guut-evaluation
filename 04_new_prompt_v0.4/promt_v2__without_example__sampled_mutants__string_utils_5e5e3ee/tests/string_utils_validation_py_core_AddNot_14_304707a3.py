from string_utils.validation import is_credit_card

def test__is_credit_card_valid_type():
    """
    Test whether the target function validates a valid credit card number with a 
    correctly specified card type. The input '4111111111111111' with card type 'VISA' 
    should return True in the baseline but will raise a KeyError in the mutant due to its 
    flawed card type checking logic.
    """
    output = is_credit_card('4111111111111111', card_type='VISA')  # a valid credit card number for VISA
    assert output == True