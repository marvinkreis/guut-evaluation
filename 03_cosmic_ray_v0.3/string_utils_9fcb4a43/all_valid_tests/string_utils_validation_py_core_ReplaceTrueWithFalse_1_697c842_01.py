from string_utils.validation import is_credit_card

def test__is_credit_card():
    """The mutant flips the return value of is_credit_card. 
    It should return True for a valid credit card number. 
    Using a known valid card number, the test ensures correctness."""
    valid_credit_card_number = '4111111111111111'
    output = is_credit_card(valid_credit_card_number)
    assert output is True, "is_credit_card must return True for valid credit card numbers"