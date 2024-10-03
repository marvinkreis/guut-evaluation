from string_utils.validation import is_credit_card

def test__is_credit_card():
    # Test valid credit card numbers (assuming these are valid Visa cards)
    valid_visa_card = "4111111111111111"
    assert is_credit_card(valid_visa_card) == True, "Expected True for valid VISA card"
    
    # Test an invalid credit card number
    invalid_card = "1234567890123456"
    assert is_credit_card(invalid_card) == False, "Expected False for invalid card number"
    
    # Test a case that would normally succeed in normal function
    valid_mastercard = "5500000000000004"
    assert is_credit_card(valid_mastercard) == True, "Expected True for valid MasterCard"

    # Test a non-numeric string
    non_numeric = "NotANumber123"
    assert is_credit_card(non_numeric) == False, "Expected False for non-numeric input"