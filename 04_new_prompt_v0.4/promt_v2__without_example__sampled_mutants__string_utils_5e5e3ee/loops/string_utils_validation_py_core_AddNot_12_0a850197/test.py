from string_utils.validation import is_credit_card

def test__is_credit_card_valid_input():
    """
    Test if the function is_credit_card accepts a valid credit card number.
    The input '4111111111111111' is a valid Visa card number and should return True.
    The mutant handles this input differently due to the logical inversion change, returning False instead.
    """
    output = is_credit_card('4111111111111111')  # Expected: True
    assert output == True