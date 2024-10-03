from string_utils.validation import is_credit_card

def test_is_credit_card():
    # This test case will check if the function correctly identifies an invalid empty credit card input.
    # The correct implementation should return False for an empty string,
    # while the mutant incorrectly returns True.
    
    # Test with an empty string input
    result = is_credit_card('')
    assert result is False, f"Expected False, but got {result}"
    
    # Also test with a known valid credit card number
    valid_card = '4111111111111111'  # Visa card number that should be valid
    assert is_credit_card(valid_card) is True, "Valid card should return True"