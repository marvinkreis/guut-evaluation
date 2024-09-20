from string_utils.validation import is_credit_card

def test__is_credit_card():
    """Mutant changes the condition to NOT match credit card regex, causing it to incorrectly validate invalid cards."""
    valid_card = "4111111111111111"
    invalid_card = "1234567812345678"
    
    assert is_credit_card(valid_card) == True, "Valid card must return True"
    assert is_credit_card(invalid_card) == False, "Invalid card must return False"