from string_utils.validation import is_credit_card

def test__is_credit_card():
    """The mutant changes the credit card validation logic, causing valid cards to return false."""
    valid_credit_card_number = '4111111111111111'
    assert is_credit_card(valid_credit_card_number, 'VISA') == True, "The function must identify valid VISA credit card"