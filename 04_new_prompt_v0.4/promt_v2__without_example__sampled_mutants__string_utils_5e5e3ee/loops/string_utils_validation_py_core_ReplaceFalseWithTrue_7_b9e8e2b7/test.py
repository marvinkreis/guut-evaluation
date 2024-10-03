from string_utils.validation import is_credit_card

def test__is_credit_card():
    """
    Test an invalid credit card number. The input '1234567890123456' is not a 
    valid credit card number, leading to a False return from the baseline but 
    True from the mutant. This test should fail when executed with the mutant.
    """
    output = is_credit_card('1234567890123456')
    assert output == False