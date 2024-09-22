from string_utils.validation import is_credit_card

def test_is_credit_card():
    # Test valid credit card numbers (pure numeric)
    assert is_credit_card('4111111111111111') == True  # Valid Visa
    assert is_credit_card('5500000000000004') == True  # Valid MasterCard
    
    # Test invalid credit card numbers
    assert is_credit_card('1234567890123456') == False  # Invalid card number
    assert is_credit_card('') == False  # Empty string
    assert is_credit_card('abcd') == False  # Non-numeric input

    # Additional tests for valid card numbers
    assert is_credit_card('4012888888881881') == True  # Another valid Visa
    assert is_credit_card('378282246310005') == True  # Valid American Express

    # Test with non-10 or non-16 digit numbers (which should return False)
    assert is_credit_card('4111111111') == False  # Too few digits
    assert is_credit_card('4111111111111111111') == False  # Too many digits

    # Test edge cases
    assert is_credit_card(None) == False  # None input

# Run the test
test_is_credit_card()