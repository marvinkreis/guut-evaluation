from string_utils.validation import is_credit_card

def test__is_credit_card():
    """
    Test the is_credit_card function with a valid credit card number. The input '4111111111111111' is known to be a valid Visa card number. 
    In the original implementation, this should return True, 
    while the mutant version will incorrectly return False due to the mutation that swaps the return values in the loop.
    """
    output = is_credit_card('4111111111111111')
    assert output == True