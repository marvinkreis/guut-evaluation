from string_utils.validation import is_credit_card

def test__is_credit_card():
    """
    Test to verify the is_credit_card function correctly identifies valid credit card numbers.
    The input '4111111111111111' is a valid Visa credit card number, which should return True.
    The mutant, however, incorrectly returns False due to the inverted logic in the implementation.
    """
    valid_credit_card = '4111111111111111'  # Example of a valid Visa card number
    output = is_credit_card(valid_credit_card) 
    assert output == True  # Expect the output to be True