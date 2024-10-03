from string_utils.validation import is_credit_card

def test__is_credit_card():
    """
    Test whether an invalid credit card number is correctly identified. 
    The input represents an invalid credit card number ('1234567890123456').
    The original function should return False because this is not a valid credit card number.
    The mutant inverts the condition to return True for invalid cards, which will cause this test to pass on the original 
    implementation but fail on the mutant.
    """
    output = is_credit_card('1234567890123456')  # An invalid credit card number
    assert output == False